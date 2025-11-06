# Vision Tower & Connector Overview

This page introduces the two main components that turn raw image patches into embeddings that the language model can consume: the vision tower and the connector.

## Vision Tower (SigLIP)

SmolVLM-256M uses a SigLIP-base vision encoder pretrained on image-text pairs. It accepts a 512×512 image and produces a grid of 32×32 patch embeddings with 768 channels. These values come from the configuration: `image_size=512`, `patch_size=16` and `vision_hidden=768` in the original config. (See [`config.json` lines 14–16](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/blob/c2bf2d847b92fbb2abe5a5b6e8825c99efcfade2/config.json#L14-L16) for these parameters.)

Key points to remember:
- The vision tower outputs a `(B, 32×32, 768)` tensor of patch features.
- Its output dimension (`vision_hidden`) matches the `hidden_size` of SigLIP (768).

## Connector

The Idefics3Connector bridges the gap between the vision tower and the language model. It performs a pixel-unshuffle (space-to-depth) operation controlled by the `pixel_shuffle_factor` and then projects the resulting features into the text hidden size.

Important configuration links:
- `pixel_shuffle_factor`: 4 (see [`config.json` line 149](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/blob/c2bf2d847b92fbb2abe5a5b6e8825c99efcfade2/config.json#L149)). This reduces the 32×32 grid to an 8×8 grid.
- `hidden_size` of the text model: 576 (see [`config.json` line 44](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/blob/c2bf2d847b92fbb2abe5a5b6e8825c99efcfade2/config.json#L44)). The connector projects each downsampled vector into this dimension.

When applied to a single 512×512 patch, the connector outputs 64 visual tokens of size 576. In the next section we will explore how the multi-patch processor uses these components to handle larger images.
