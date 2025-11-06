# Multi-Patch Overview

## Why Multi-Patch?

SmolVLM’s default vision tower is SigLIP-base, which operates on 512×512 inputs. To handle larger images, the model divides high-resolution images into multiple 512×512 patches. Each patch is resized and passed through the shared vision encoder. The number of visual tokens produced from each patch and the amount of downsampling are specified in the configuration:

- The **text hidden size** is 576, as set in [`config.json` line 44](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/blob/c2bf2d847b92fbb2abe5a5b6e8825c99efcfade2/config.json#L44). This defines the dimension of the projected token embeddings for the language model.
- The **number of visual tokens per patch** is defined by `resampler_n_latents` in the configuration, which is 64, as seen at [`config.json` lines 127–131](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/blob/c2bf2d847b92fbb2abe5a5b6e8825c99efcfade2/config.json#L127-L131). After pixel unshuffling and linear projection, each 512×512 patch produces 64 visual tokens.
- The **pixel shuffle factor** is 4, specified at [`config.json` line 149](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/blob/c2bf2d847b92fbb2abe5a5b6e8825c99efcfade2/config.json#L149). This means the 32×32 patch grid produced by SigLIP is downsampled to an 8×8 grid (since 32 / 4 = 8) before projection.

These parameters come from the original Hugging Face `config.json` for SmolVLM‑256M. We will refer back to these settings as we explore the implementation.

## High‑Level Pipeline

1. **Patch extraction**: A high‑resolution image is split into N non‑overlapping 512×512 patches. The value of N depends on the original image resolution; for example, an image sized 2048×2048 yields 16 patches.
2. **Vision encoding**: Each patch is fed through the SigLIP vision tower, yielding a `(B, 32×32, 768)` tensor of patch embeddings (B = batch size).
3. **Connector downsampling**: The `Idefics3Connector` performs pixel unshuffle with scale factor 4 to reduce the 32×32 patch grid to an 8×8 grid and projects the resulting features to the language model’s hidden size (576). The output is `(B, 64, 576)` for each patch.
4. **Token concatenation**: The outputs of all patches are concatenated along the sequence dimension to form a `(B, N×64, 576)` tensor. This is then packed into the text model’s input embeddings at positions indicated by `<image>` tokens.

This overview sets the stage for understanding the `MultiPatchVisionProcessor` class, which implements this logic in code. In subsequent pages we will walk through the class structure and functions line by line.
