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
- Resize and center‑crop the image to the specified `image_size` and convert it to a normalized tensor.  
- Insert a `<image>` placeholder token into the text sequence to represent the image.  
- Return a dictionary with `input_ids`, `pixel_values`, `attention_mask` and other fields.  
The SmolVLM model’s `forward` method then processes `pixel_values` through the vision tower and stitches the resulting embeddings into the position of the `<image>` token during the forward or generation call.  
## Key parameters  
The following configuration values determine how images are preprocessed:  
- `image_size` – the side length used to resize images before patch extraction. In the SmolVLM‑256M‑Instruct checkpoint this is 512 ([huggingface.co](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/blob/c2bf2d847b92fbb2abe5a5b6e8825c99efcfade2/config.json)).  
- `patch_size` – the size of each patch for the SigLIP tower; the 256M model uses a 16×16 patch size ([huggingface.co](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/blob/c2bf2d847b92fbb2abe5a5b6e8825c99efcfade2/config.json)).  
- `num_channels` – usually 3 for RGB images.  
## Next steps  
In later pages we will delve into the actual preprocessing functions in FMS (`preprocessing.py`) and the Hugging Face `SmolVLMProcessor`, showing code excerpts and linking to the relevant lines in both repositories. We will also explore how multi-image prompts are split into tiles and how the processor inserts the `<image>` token.  
### Hugging Face references  
The Hugging Face `SmolVLMProcessor` and `SmolVLMImageProcessor` classes implement the preprocessing pipeline for SmolVLM models. The `SmolVLMProcessor` wraps a LLaMA tokenizer and SmolVLM image processor into a single processor and inherits all functionalities of `SmolVLMImageProcessor` and `SmolVLMTokenizerFast`. In the processor’s `__call__` method, images are resized and center‑cropped to the configured `image_size`, normalized, and returned alongside the tokenized text in a dictionary containing `pixel_values`, `input_ids`, and `attention_mask`. Relevant files with commit‑specific permalinks include:  
- **SmolVLMProcessor**: [`processing_smolvlm.py`](https://github.com/huggingface/transformers/blob/f5630f9/src/transformers/models/smolvlm/processing_smolvlm.py)  
- **SmolVLMImageProcessor**: [`image_processing_smolvlm.py`](https://github.com/huggingface/transformers/blob/89a4115/src/transformers/models/smolvlm/image_processing_smolvlm.py)  
- **SmolVLM configuration**: [`configuration_smolvlm.py`](https://github.com/huggingface/transformers/blob/163601c/src/transformers/models/smolvlm/configuration_smolvlm.py)  
- **SigLIP vision configuration**: [`configuration_siglip.py`](https://github.com/huggingface/transformers/blob/d1c6310/src/transformers/models/siglip/configuration_siglip.py)  
- **SigLIP vision model**: [`modeling_siglip.py`](https://github.com/huggingface/transformers/blob/6f6095e/src/transformers/models/siglip/modeling_siglip.py)  
  
### Planned FMS implementation  
To align the Foundation Model Stack implementation with Hugging Face, we plan to extend our preprocessing helper to provide similar functionality:  
- **Resizing & normalization:** Provide functions to resize images to the configured `image_size` while preserving aspect ratio, optionally center-crop if required, and normalize pixel values.  
- **Insert `<image>` placeholder:** Insert a `<image>` placeholder token into the token sequence with the correct `image_seq_len` based on the patch size, matching the Hugging Face processor.  
- **Return processed inputs:** Return a dictionary containing the processed `pixel_values` as well as the original text tokens and attention mask so that the vision tower and language model can splice the image embeddings into the token sequence during forward and generation calls.  
This design will ensure that users can swap between the Hugging Face and FMS pipelines without changing their downstream code. 
