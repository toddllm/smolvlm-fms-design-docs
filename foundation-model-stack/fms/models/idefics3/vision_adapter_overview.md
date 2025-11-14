# Vision Adapter Overview  
 
This page describes the role of the vision adapter in the SmolVLM Foundation Model Stack implementation.  
 
## Purpose  
 
The vision adapter acts as a wrapper around the SigLIP vision tower to produce embeddings compatible with the text backbone. It bridges between the raw image tensor and the connector that projects features into the text hidden dimension.  
 
## Key points  
 
- Loads a pre‑trained SigLIP model (e.g. base‑patch16‑512) to produce patch embeddings.  
- Exposes a `forward` method that takes a batch of pixel values and returns a tensor of shape `(B, N, vision_hidden)`, where `N` is the number of patches per image.  
- Provides a `to_hf_api` or similar method so that the adapter can be converted into a Hugging Face–style vision model.  
- May handle resizing and normalization consistent with the preprocessing helper.  
 
## Detailed design  
 
To make the adapter easy to integrate with the rest of the model stack, we will implement a lightweight `VisionTowerAdapter` class. Internally it wraps the vision tower (e.g. a SigLIP or other image encoder) but hides the complexity behind a standard interface. The key design elements include:  
 
- **Initialization** – The adapter accepts the inner vision tower module and optional normalization callable. It also stores the expected number of output patches and hidden dimension. During initialization, parameters of the inner tower are frozen to ensure the vision encoder remains unchanged during text‑only fine‑tuning.  
 
```python
class VisionTowerAdapter(nn.Module):
    """
    Wraps the SmolVLM‑bundled vision tower to provide a consistent interface.
    """
    def __init__(
        self,
        inner_tower: nn.Module,
        normalize: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        expected_num_patches: int = 1024,
        expected_hidden_dim: int = 768,
    ):
        super().__init__()
        self.inner = inner_tower.eval()          # freeze vision tower
        self.normalize = normalize
        self.expected_num_patches = expected_num_patches
        self.expected_hidden_dim = expected_hidden_dim

        for param in self.inner.parameters():
            param.requires_grad = False
```  
 
- **Forward method** – The adapter's `forward` method normalizes input images if a normalization function is provided, runs them through the wrapped vision tower, and then verifies that the returned tensor has shape `(B, expected_num_patches, expected_hidden_dim)`. This check enforces that downstream components receive exactly the expected patch embeddings.  
 
```python
@torch.no_grad()
def forward(self, images: torch.Tensor) -> torch.Tensor:
    """
    Process images through vision tower and return patch embeddings.
    Args:
        images: input images (B, 3, 512, 512) scaled to [0, 1]
    Returns:
        patch embeddings (B, 1024, 768)
    Raises:
        AssertionError if the output shape does not match expectations.
    """
    x = self.normalize(images) if self.normalize else images
    feats = self.inner(x)
`` # handle different output formats (dict vs tensor)
    if isinstance(feats, dict):
        if "last_hidden_state" in feats:
            feats = feats["last_hidden_state"]
        elif "pooler_output" in feats:
            raise ValueError("Vision tower returned pooler_output, need patch features")
        else:
            raise ValueError(f"Unknown vision tower output: {feats.keys()}")
    assert feats.dim() == 3
    assert feats.size(1) == self.expected_num_patches
    assert feats.size(2) == self.expected_hidden_dim
    return feats
```  
 
- **Normalization helper** – We provide a utility `create_normalize_fn(mean, std, device)` that returns a callable to normalize images using pre‑computed per‑channel means and standard deviations. This helper can be used to quickly build a normalization function from typical dataset statistics (e.g. ImageNet).  
 
- **Checkpoint extraction** – In order to easily work with different variants of SmolVLM checkpoints, we include an `extract_vision_tower_from_checkpoint` function. It scans common attribute names (e.g. `vision_tower`, `vision`, `vision_model`, `visual`) either at the top level of the checkpoint, inside a `.model` attribute, or in a state‑dictionary, and returns the vision tower module. This helper ensures that the adapter can robustly find the vision encoder even if the checkpoint layout varies.  
 
By defining the adapter in this way, we provide a stable and well‑documented boundary between the vision tower and the rest of the model stack. Future implementations may extend the adapter to support additional vision backbones, resizing strategies or integration with Hugging Face model APIs, but the core contract remains the same: given a batch of images, produce a tensor of patch embeddings with a known shape.  
 
## Next steps  
 
In future iterations we will link to the implementation of `vision_adapter.py` in the FMS repo and compare it with the vision tower code in `modeling_smolvlm.py` from Hugging Face. We will also highlight any differences in API or behavior.
