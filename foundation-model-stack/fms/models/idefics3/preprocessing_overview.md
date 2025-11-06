# Image Preprocessing Overview

This page discusses how images are prepared before being passed to the vision tower in SmolVLM models, comparing the Foundation Model Stack (FMS) implementation to the Hugging Face Transformers version.

## FMS preprocessing

In the FMS implementation, the logic for resizing and normalizing images lives inside a preprocessing helper (for example, `preprocessing.py`). When `prepare_inputs` is called, the pre-hook will:

- Resize the input image to the configured `image_size` (e.g. 512) while maintaining aspect ratio.
- Divide the image into patches of size `patch_size` (e.g. 16) for the SigLIP vision tower.
- Normalize pixel values and apply pixel unshuffle if required by the connector.
- If the image is larger than the supported size, split it into multiple 512×512 tiles and process each patch separately (see the multi-patch overview page).

These steps run before the text model’s forward call so that the inputs contain image embeddings ready to be spliced into the token sequence.

## Transformers preprocessing

Hugging Face provides a separate processor class (such as `SmolVLMProcessor`) that handles image preprocessing. The processor will:

- Resize and center-crop the image to the specified `image_size` and convert it to a normalized tensor.
- Insert a `<image>` placeholder token into the text sequence to represent the image.
- Return a dictionary with `input_ids`, `pixel_values`, `attention_mask` and other fields.

The SmolVLM model’s `forward` method then processes `pixel_values` through the vision tower and stitches the resulting embeddings into the position of the `<image>` token during the forward or generation call.

## Key parameters

The following configuration values determine how images are preprocessed:

- `image_size` – the side length used to resize images before patch extraction. In the SmolVLM-256M-Instruct checkpoint this is 512 ([huggingface.co](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/blob/c2bf2d847b92fbb2abe5a5b6e8825c99efcfade2/config.json#:~:text=,4)).
- `patch_size` – the size of each patch for the SigLIP tower; the 256M model uses a 16×16 patch size ([huggingface.co](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/blob/c2bf2d847b92fbb2abe5a5b6e8825c99efcfade2/config.json#:~:text=,4)).
- `num_channels` – usually 3 for RGB images.

## Next steps

In later pages we will delve into the actual preprocessing functions in FMS (`preprocessing.py`) and the Hugging Face `SmolVLMProcessor`, showing code excerpts and linking to the relevant lines in both repositories. We will also explore how multi-image prompts are split into tiles and how the processor inserts the `<image>` token.
