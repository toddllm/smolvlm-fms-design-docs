# Multi-Patch Processor Factory & Usage

This page will explain the helper function `create_multi_patch_processor` and demonstrate how to instantiate and use the multi‑patch vision processor.

## Purpose of the Factory

The factory wraps the configuration details and returns a ready‑to‑use `MultiPatchVisionProcessor`. It chooses the appropriate vision encoder and connector, and sets default parameters such as the number of tokens per patch.

## Using the Factory

Below is a minimal example of how you might call the factory once the necessary modules are available:

```python
# pseudocode
vision_encoder = build_vision_encoder_from_config(...)
connector = Idefics3Connector(...)
processor = create_multi_patch_processor(
    vision_encoder=vision_encoder,
    connector=connector,
    tokens_per_patch=64,
    use_hf_connector=False,
)

# Given a list of image patches (each of shape [B, C, 512, 512])
embeddings, stats = processor.process_patches(patches)
```

In later iterations we will reference specific lines in the original `multi_patch.py` file and configuration values to explain exactly how this factory sets up the processor.
