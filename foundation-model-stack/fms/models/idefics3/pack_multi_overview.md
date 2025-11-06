# Multi-Image Packing Overview

This page introduces the multi-image packing logic used in the SmolVLM Foundation Model Stack for inputs containing multiple images.

## Motivation

When prompts include more than one `<image>` placeholder, we need to pack the visual tokens from each image into the text sequence without exceeding the maximum token length. The multi-image packer extends the single-image packer to iterate over all image slots and insert a fixed-length span of visual tokens for each.

## Key ideas

- **Multiple `<image>` tokens**: The packer scans the `input_ids` for every occurrence of the model's `image_token_id` and ensures there is a corresponding image in the batch.
- **Per-image span length**: Each image contributes a fixed `image_span_len` number of tokens, determined by the configuration's `resampler_n_latents` field (for example, 64 for the SmolVLM-256M-Instruct checkpoint).
- **Splicing order**: Visual tokens are spliced into the sequence in the same order as the `<image>` tokens appear in the prompt.

## FMS vs HF

In our FMS integration, packing happens in the `prepare_inputs` pre-hook before the text model's forward call. The multi-image packer uses a helper function that loops over each image slot and inserts the corresponding visual tokens into the sequence.

In the Hugging Face Transformers implementation, images and their embeddings are processed during the model's forward/generate call, and the model can handle multiple images transparently, relying on the processor/template to prepare the inputs.
