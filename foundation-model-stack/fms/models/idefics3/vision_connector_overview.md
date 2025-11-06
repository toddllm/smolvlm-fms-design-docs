# Vision Tower & Connector Overview

This page introduces the two main components that turn raw image patches into embeddings that the language model can consume: the vision tower and the connector.

## Vision Tower (SigLIP)

SmolVLM-256M uses a SigLIP-base vision encoder trained on image–text pairs. It accepts a 512×512 image and produces a grid of 32×32 patch embeddings with 768 channels. These values come from the configuration: image_size=512, patch_size=16 and vision_hidden=768 in the original config.

**Key points:**
- The vision tower outputs a tensor of shape (B, 32×32, 768) representing 1,024 patch features.
- When the pixel_shuffle_factor is 4, these patches will be downsampled to an 8×8 grid later by the connector.

## LlavaNext Vision Tower

FMS also includes a vision tower for the LlavaNext model. In llava_next.py the LlavaNextConfig sets up a SiglipVisionConfig with hidden_size=1152, image_size=384 and patch_size=14. This configuration produces a different number of patches (depending on the image resolution) and uses a multimodal projector to align vision features with a Granite text backbone. When porting LlavaNext to FMS, adjust the connector scale and hidden sizes according to these values.

## Connector

The Idefics3Connector bridges the gap between the vision tower and the language model. It performs a pixel-unshuffle (space-to-depth) operation controlled by the pixel_shuffle_factor and then projects the resulting features into the text hidden size.

Important configuration links:
- pixel_shuffle_factor = 4, which reduces a 32×32 grid to an 8×8 grid.
- hidden_size of the text model = 576; the connector projects each downsampled vector into this dimension.

When
For reference, the SmolVLM-256M-Instruct configuration specifies `image_size=512`, `patch_size=16` and `vision_hidden=768` on lines 14–16 of its `config.json`: https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/blob/c2bf2d847b92fbb2abe5a5b6e8825c99efcfade2/config.json#L14-L16. It sets `pixel_shuffle_factor=4` on line 149 (same link) and the text model's `hidden_size=576` on line 44. The LlavaNext vision tower uses `hidden_size=1152`, `image_size=384` and `patch_size=14`, as defined in the FMS `llava_next.py` source: https://github.com/foundation-model-stack/foundation-model-stack/blob/main/fms/models/llava_next.py#L43-L52.
 applied to a single 512×512 patch, the connector outputs 64 visual tokens of size 576. In the next section we will explore how the multi-patch processor uses these components to handle larger images.
