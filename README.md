# SmolVLM FMS Design Docs

This repository contains design documentation for the SmolVLM variant of the foundation-model-stack. The goal is to teach the implementation from first principles, one file at a time, explaining the reasoning behind each component.

## Overview
The SmolVLM model combines a SigLIP vision tower with a decoder-only text model. Images are processed as patches, projected via a connector into the text dimension, and then packed into the text sequence. The components we cover include:

- Vision tower – a SigLIP encoder that outputs patch features from 512x512 images with 16x16 patch size (32x32 grid). For SmolVLM this produces 1024 patches of dimension 768.
- Connector – the Idefics3Connector performs pixel-unshuffle on the patch grid (scale 4) and projects the concatenated features into the text hidden dimension (576). This reduces the 32x32 patch grid to 8x8 latents, for a total of 64 visual tokens.
- Hidden align – optional linear layer to map connector output dimension to the text model hidden size when they differ.
- Packer – function that finds `<image>` placeholder tokens in the text input IDs and splices the 64 visual tokens into the sequence. The placeholder's token ID and number of visual tokens should be read from the config, rather than hard-coded.
- Text backbone – a decoder-only language model (for example Llama 2 7B). The generate method should support both generate_from_embeds and generate(inputs_embeds=...) for models that provide these signatures.

## Design goals
- Parameter alignment: do not hard-code values like image_token_id, connector_scale, image_span_len, or vision_hidden. Instead, read these from the Hugging Face config.json for the chosen checkpoint. For SmolVLM these are image_token_id=49190, pixel_shuffle_factor=4, resampler_n_latents=64, and text_hidden_size=576, as found in the official config.
- Support multiple vision towers: the foundation-model-stack can swap in different vision encoders (for example NaViT for the Idefics3-8B model). Document how to adjust vision_hidden, image_size, patch_size, and connector_scale when using alternative towers.
- Robust generation: implement fallback logic so that if the underlying LLM does not implement generate_from_embeds, it falls back to generate(inputs_embeds=...), and finally to providing input_ids if necessary.

## Embedding the code
Below is an excerpt from foundation-model-stack/fms/models/idefics3/model.py showing the HiddenAlign class:

```
class HiddenAlign(nn.Module):
    """
    Optional projection layer to align connector output dimension
    to text model hidden dimension if they differ.
    """
    def __init__(self, src_dim: int, dst_dim: int):
        super().__init__()
        self.need_projection = (src_dim != dst_dim)
        self.proj = nn.Linear(src_dim, dst_dim, bias=False) if self.need_projection else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T_img, src_dim) -> (B, T_img, dst_dim)"""
        return self.proj(x)
```

Additional files in this repo will dive deeper into each component, providing code snippets and links to the relevant sections of the configuration. See foundation-model-stack/fms/models/idefics3/model.py for the full implementation.

---

This README serves as the starting point for the SmolVLM FMS design docs. Future markdown files will explore connector.py, pack.py, and the adapter classes, along with diagrams and examples explaining the reasoning behind each design decision.
