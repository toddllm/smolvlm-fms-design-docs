# Vision Adapter Overview

This page describes the role of the vision adapter in the SmolVLM Foundation Model Stack implementation.

## Purpose

The vision adapter acts as a wrapper around the SigLIP vision tower to produce embeddings compatible with the text backbone. It bridges between the raw image tensor and the connector that projects features into the text hidden dimension.

## Key points

- Loads a pre-trained SigLIP model (e.g. base-patch16-512) to produce patch embeddings.
- Exposes a `forward` method that takes a batch of pixel values and returns a tensor of shape `(B, N, vision_hidden)`, where `N` is the number of patches per image.
- Provides a `to_hf_api` or similar method so that the adapter can be converted into a Hugging Face-style vision model.
- May handle resizing and normalization consistent with the preprocessing helper.

## Next steps

In future iterations we will link to the implementation of `vision_adapter.py` in the FMS repo and compare it with the vision tower code in `modeling_smolvlm.py` from Hugging Face. We will also highlight any differences in API or behavior.
