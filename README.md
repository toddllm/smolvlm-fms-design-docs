# SmolVLM FMS Design Docs

## Overview
Describe FMS adaptation: the vision tower (SigLIP or LlavaNext), connector, hidden align, packer, text backbone.

- **Vision tower** – extracts image patch features using SigLIP or LlavaNext towers.
- **Connector** – down-samples and projects vision features to text hidden size.
- **Hidden align** – optional projection layer to align connector output with text hidden dimension.
- **Packer** – inserts visual token spans into the text sequence at `<image>` placeholders.
- **Text backbone** – underlying language model that consumes combined embeddings.

## Design goals
- **Parameter alignment** – Do not hard-code values like `image_token_id`, `pixel_shuffle_factor`, `resampler_n_latents`, or `vision_hidden`. Instead, read these from the Hugging Face config for the chosen checkpoint. For SmolVLM‑256M‑Instruct these are `image_token_id=49190`, `pixel_shuffle_factor=4`, `resampler_n_latents=64`, and `text_hidden=576`.
- **Support multiple vision towers** – FMS can swap in different vision encoders (e.g., SigLIP, LlavaNext). Document how to adjust `vision_hidden`, `image_size`, `patch_size`, and `pixel_shuffle_factor` when using alternative towers.
- **Robust generation** – Implement fallback logic so that if the underlying LLM does not implement `generate_from_embeds`, it falls back to `generate(inputs_embeds=...)`, and finally to providing `input_ids` if necessary.

## Embedding the code

```python
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

## Documentation index
- **[Porting overview](foundation-model-stack/fms/models/idefics3/porting_overview.md)** – High-level discussion of why reimplement SmolVLM in FMS and patterns observed when porting from Hugging Face.
- **[FMS vs Transformers overview](foundation-model-stack/fms/models/idefics3/fms_vs_transformers_overview.md)** – Compares design goals and modular structure of FMS with Transformers, including optimized inference/training and adapter methods.
- **[Vision tower & connector overview](foundation-model-stack/fms/models/idefics3/vision_connector_overview.md)** – Explains the SigLIP and LlavaNext vision towers, how the connector down-samples and projects image patches, and links to both FMS and HF implementations.
- **[Vision adapter overview](foundation-model-stack/fms/models/idefics3/vision_adapter_overview.md)** – Describes the purpose of the vision adapter wrapper around the SigLIP tower and how it interfaces with the FMS model, including `to_hf_api`.
- **[HF adapter overview](foundation-model-stack/fms/models/idefics3/hf_adapter_overview.md)** – Introduces the Hugging Face adapter pattern in FMS and shows how an FMS model can be wrapped to behave like an HF model.
- **[Preprocessing overview](foundation-model-stack/fms/models/idefics3/preprocessing_overview.md)** – Summarizes how images are resized and normalized in FMS and HF before feeding into the vision tower, highlighting differences in the pre-hook and forward flows.
- **[Pack overview](foundation-model-stack/fms/models/idefics3/pack_overview.md)** – Describes the packing of visual tokens into the text sequence for a single image, explaining configuration values like `image_token_id` and `resampler_n_latents`.
- **[Pack multi overview](foundation-model-stack/fms/models/idefics3/pack_multi_overview.md)** – Extends pack overview to multiple images; explains how multiple image spans are handled and spliced into text.
- **[Multi-patch overview](foundation-model-stack/fms/models/idefics3/multi_patch_overview.md)** – Explains the rationale behind the multi-patch processor for high-resolution images and describes the high-level pipeline.
- **[Multi-patch class](foundation-model-stack/fms/models/idefics3/multi_patch_class.md)** – Provides a skeleton of the `MultiPatchVisionProcessor` class and its key attributes.
- **[Process patches walkthrough](foundation-model-stack/fms/models/idefics3/process_patches_walkthrough.md)** – Walks through the `process_patches` method step by step, detailing how each patch is processed and concatenated.
- **[Multi-patch factory usage](foundation-model-stack/fms/models/idefics3/multi_patch_factory_usage.md)** – Shows how to instantiate the multi-patch processor and integrate it into a workflow.

This README serves as the starting point for the SmolVLM FMS design docs. Future markdown files will explore `connector.py`, `pack.py`, and the adapter classes, along with diagrams and examples explaining the reasoning behind each design decision.
