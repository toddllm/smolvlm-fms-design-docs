# Vision families: Docling vs Mistral  

## Shared architecture patterns  

- Both use a ViT-like vision backbone to convert image patches into latent tokens that feed into a language decoder.  
- Each uses a projector/connector to map vision hidden dimension to text hidden dimension.  
- Visual tokens are packed into the language token sequence as placeholders before being consumed by the decoder.  
- Multi-image support is built in; they can handle multiple pages or images.  

## Key differences  

### Preprocessing & geometry  

Docling (SigLIP2-based) uses a fixed canvas (512×512) with patch size 16 → 32×32 grid of patches. The pixel-shuffle connector down-samples to an 8×8 grid (64 latents) before projection into the text hidden size 576. This ensures a fixed number of visual tokens per image. The design doc notes that image_size=512, patch_size=16, pixel_shuffle_factor=4 and resampler_n_latents=64 in the config.  

Pixtral (Mistral’s vision track) supports arbitrary resolutions via 2D RoPE and does not pad to a canonical square. The number of vision tokens is derived from the input image size. There is no explicit pixel-shuffle or perceiver resampler described, so token count is variable.  

### Token compaction strategy  

Docling uses a pixel-shuffle (space-to-depth) connector, inherited from Idefics-3, to improve OCR fidelity. The connector takes the 32×32 patch grid and groups pixels with factor 4 to produce 8×8 features (64 latents) for the language model.  

Pixtral presumably uses the vision encoder directly (with 2D RoPE) and a projector; token counts are not compacted via pixel-shuffle; they remain proportional to image resolution.  

### Deriving visual token count  

Docling: The number of visual tokens per image equals the `resampler_n_latents` parameter (64 for SmolVLM‑Docling) derived from the HF config. The porting overview emphasises not to hard‑code `pixel_shuffle_factor` or token counts; these should be read from the config. When using other image sizes or patch sizes, compute the span length as `(img_size / patch_size / pixel_shuffle_factor)^2`.  

Pixtral: The number of tokens is not predetermined. For arbitrary size images, measure the encoder output length or use a provided field to allocate placeholders.  

### Placeholder semantics & generation API  

Docling inherits Idefics-style requirement to provide `input_ids` on the first generation call because not all frameworks support `inputs_embeds` for multimodal models. Pack placeholders equal to the number of vision tokens (resampler_n_latents) into the input sequence.  

Pixtral uses a dedicated vision encoder and Mistral decoder; if using frameworks like vLLM, the rule is to pack placeholder tokens equal to the encoder output length. Since vLLM’s docs emphasise this generic contract, the packer should check that #images matches #placeholders.  

### Declared internals  

Docling uses a SigLIP2-base-patch16-512 vision tower with pixel-shuffle connector; config values include `vision_hidden=768`, `text_hidden=576`, `pixel_shuffle_factor=4` and `resampler_n_latents=64`. These correspond to Granite-Docling-258M or SmolVLM weights.  

Mistral’s Small‑3.1 24B family provides long context (128k) and multi-image support but does not publicly specify patch size or compactor. It is thus considered an opaque or black-box vision module; treat details as unknown.  

## Practical checklist  

Create a table summarizing the key implementation details:  

| Aspect | Docling (e.g., SigLIP2/Granite‑Docling) | Mistral (Pixtral/Small‑3.1) |  
| --- | --- | --- |  
| Input canvas policy | Fixed 512×512 square; patch size 16; normalization using SigLIP mean/std | Arbitrary resolution via 2D RoPE (no resizing/padding) |  
| Patch geometry & initial tokens | 32×32 grid of patches (1024 tokens) | Dependent on image size; unspecified patch size |  
| Compaction path | Pixel-shuffle (space-to-depth) with factor 4 to yield 8×8 = 64 latents | No public compactor; token count derived from image resolution |  
| Vision → Text projection | Linear/MLP projection from vision_hidden=768 to text_hidden=576; config-driven | Projector maps vision hidden dims to Mistral hidden; size unspecified |  
| Visual token count (T_img) | Derived from config `resampler_n_latents` (64 for SmolVLM/Docling) | Derived from encoder output length; variable |  
| Placeholder packing | Insert placeholders equal to T_img before text tokens; require `input_ids` on first generation | Pack placeholders equal to encoder output length; frameworks like vLLM follow generic contract |  
| Multi-image support | Yes; per-page encode→project→pack | Yes; supports many images and long context |  

## Summary  

Docling and Mistral's vision-language models share the high-level pattern of a vision encoder feeding a projection layer and then a decoder-only language model. They differ mainly in how they handle image preprocessing and the compaction of visual tokens. Docling's SigLIP2 tower expects a fixed 512×512 canvas and uses a pixel-shuffle connector that deterministically produces 64 latent tokens per image, whereas Pixtral supports arbitrary image sizes and has a flexible number of visual tokens. When integrating these models into systems or building connectors, avoid hard coding token counts or placeholder positions; instead read the configuration (e.g., `resampler_n_latents`, `pixel_shuffle_factor`) or measure encoder outputs to derive the appropriate number of placeholders.
