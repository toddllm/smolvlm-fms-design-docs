# Packing & Splicing Overview  

## Why packing?  
- In multimodal LLMs like SmolVLM, image embeddings need to be inserted into the sequence of text tokens at positions corresponding to `<image>` placeholders.  
- The number of visual tokens per image is determined by the resampler latent count (e.g., 64) from the config ([huggingface.co](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/blob/c2bf2d847b92fbb2abe5a5b6e8825c99efcfade2/config.json#L127#:~:text=,true)).  
- The token id for the `<image>` placeholder is defined in the tokenizer config (for SmolVLM it is 49190) and may differ across models.  

## Packers in FMS  
- `pack.py` defines a `pack_image_embeddings` function that scans the `input_ids` for occurrences of the `image_token_id` and replaces each occurrence with the provided image embeddings.  
- It asserts that exactly `image_span_len` tokens (e.g., 64) will be inserted at each placeholder; this ensures the shapes align.  
- `pack_multi.py` extends this to multi-patch scenarios by repeating the embedding packing for each patch placeholder.  

## Packing in HF  
- In the Hugging Face implementation, the packing of image embeddings happens inside the model's `forward` or `prepare_inputs_for_generation` method. The model detects `<image>` token ids in `input_ids`, processes the images through its vision tower, and concatenates the resulting embeddings with the text embeddings during the forward pass.  
- The difference is that FMS uses a standalone pack function executed in a pre-hook before the text model's forward call, whereas HF does the packing during the forward call itself.  

## Key parameters  
- **`image_token_id`** — the integer id representing the `<image>` placeholder (e.g., 49190 for SmolVLM).  
- **`image_span_len`** — number of visual tokens per image (64 for SmolVLM as set by `resampler_n_latents` ([huggingface.co](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/blob/c2bf2d847b92fbb2abe5a5b6e8825c99efcfade2/config.json#L127#:~:text=,true))).  
- When adding multi-patch support, `image_span_len` may remain the same but repeated for each patch.
