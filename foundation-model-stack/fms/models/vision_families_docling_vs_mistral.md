# Vision families: Docling vs Mistral

> **Purpose:** Side-by-side, code-anchored notes for ports and packer/debug work.  
> **TODO:** add permalinks with `#Lx-Ly` to each reference after opening the file in GitHub.

---

## Shared architecture patterns

- ViT-family vision encoder, then projector/connector, then decoder-only LLM; visual spans are **packed** with text.
  - FMS: `foundation-model-stack/fms/models/idefics3/vision_connector_overview.md#Lx-Ly`
  - FMS vs HF: `foundation-model-stack/fms/models/idefics3/fms_vs_transformers_overview.md#Lx-Ly`
  - HF SmolVLM:
    - `transformers/src/transformers/models/smolvlm/configuration_smolvlm.py#Lx-Ly`
    - `transformers/src/transformers/models/smolvlm/modeling_smolvlm.py#Lx-Ly`

- Packing: produced feature length must match the count of `<image>` placeholders.
  - FMS packer: `foundation-model-stack/fms/models/idefics3/pack_overview.md#Lx-Ly`
  - Multi-image/pages: `foundation-model-stack/fms/models/idefics3/pack_multi_overview.md#Lx-Ly`

---

## Where they align (ops you can count on)

### Vision backbone is ViT-family
- **Docling:** SigLIP(2) tower, patch-16, fixed canvas.
  - FMS: `foundation-model-stack/fms/models/idefics3/vision_connector_overview.md#Lx-Ly`
  - Preproc: `foundation-model-stack/fms/models/idefics3/preprocessing_overview.md#Lx-Ly`
  - HF SigLIP config: `transformers/src/transformers/models/siglip/configuration_siglip.py#Lx-Ly`

- **Mistral (Pixtral / Small-3.1 Vision):** ViT-like encoder → projector → Mistral decoder.
  - Pixtral docs/card: **TODO add permalink**

### Projector/Connector maps vision-hidden to text-hidden
- **Docling:** Idefics-3-style connector with **pixel-shuffle**, then linear/MLP to text hidden.
  - FMS: `foundation-model-stack/fms/models/idefics3/vision_connector_overview.md#Lx-Ly`
  - HF SmolVLM usage: `transformers/src/transformers/models/smolvlm/modeling_smolvlm.py#Lx-Ly`

- **Pixtral:** explicit encoder → projector → decoder split.
  - Pixtral docs/card: **TODO add permalink**

### Pack visual tokens into the language sequence
- Both interleave visual spans with text and decode with a decoder-only LLM.
  - FMS: `foundation-model-stack/fms/models/idefics3/pack_overview.md#Lx-Ly`
  - vLLM placeholder rule: **TODO add permalink**

### Multi-image / multi-page support
- **Docling:** repeat **preprocess → encode → project → pack** per page.
  - FMS: `foundation-model-stack/fms/models/idefics3/pack_multi_overview.md#Lx-Ly`

- **Mistral:** long context and multi-image.
  - Pixtral docs/card: **TODO add permalink**

---

## Where they diverge (and why it matters)

### Preprocessing & geometry
- **Docling (SigLIP2):** fixed 512 canvas, patch-16 → 32×32 grid; SigLIP mean/std.
  - FMS: `foundation-model-stack/fms/models/idefics3/preprocessing_overview.md#Lx-Ly`
  - FMS: `foundation-model-stack/fms/models/idefics3/vision_connector_overview.md#Lx-Ly`
  - HF image processor: `transformers/src/transformers/models/smolvlm/image_processing_smolvlm.py#Lx-Ly`

- **Pixtral:** arbitrary resolution/aspect via **2D RoPE**; token count varies with input size.
  - Pixtral docs/card: **TODO add permalink**

### Token compaction strategy (resampler vs pixel-shuffle vs none)
- **Docling / Idefics-3:** **pixel-shuffle (space-to-depth)** replaces Perceiver resampler.  
  With 512/16 (32×32) and factor 4 → 8×8 = **64** tokens.
  - FMS: `foundation-model-stack/fms/models/idefics3/vision_connector_overview.md#Lx-Ly` (look for `pixel_shuffle_factor=4`, `resampler_n_latents=64`)
  - HF SmolVLM config: `transformers/src/transformers/models/smolvlm/configuration_smolvlm.py#Lx-Ly`

- **Pixtral:** encoder + 2D RoPE + projector; **variable** token count; no pixel-shuffle noted publicly.
  - Pixtral docs/card: **TODO add permalink**

### How many visual tokens per image (and where to read it)
- **Docling:** often **64** for SigLIP2 512/16 + shuffle 4.  
  **Do not hard-code**; read from config or produced shapes.
  - FMS porting: `foundation-model-stack/fms/models/idefics3/porting_overview.md#Lx-Ly`
  - HF config fields: `transformers/src/transformers/models/smolvlm/configuration_smolvlm.py#Lx-Ly`

- **Pixtral:** no universal span; measure encoder output or use provided field.
  - Pixtral docs/card: **TODO add permalink**

### Placeholders & generation entry points
- **Idefics/SmolVLM-style:** may require `input_ids` with `<image>` placeholders on first `generate`; if multimodal `inputs_embeds` is unsupported, fall back to `input_ids`.
  - FMS: `foundation-model-stack/fms/models/idefics3/pack_overview.md#Lx-Ly`
  - HF SmolVLM: `transformers/src/transformers/models/smolvlm/modeling_smolvlm.py#Lx-Ly` (search `generate`, `inputs_embeds`, `input_ids`, `image_token_id`)

- **vLLM contract:** `#placeholders == feature length`; validate `#images == #placeholders`.
  - vLLM docs: **TODO add permalink**

---

## Per-model checklist

- **Inputs / preprocessing**  
  Docling: fixed 512; SigLIP mean/std.  
  Refs: `foundation-model-stack/fms/models/idefics3/preprocessing_overview.md#Lx-Ly`  
  Pixtral: arbitrary res; **TODO add permalink**

- **Patch geometry**  
  Docling: 512/16 → 32×32 grid.  
  Ref: `foundation-model-stack/fms/models/idefics3/vision_connector_overview.md#Lx-Ly`

- **Compaction path**  
  Docling: pixel-shuffle projector (OCR-friendly).  
  Ref: `foundation-model-stack/fms/models/idefics3/vision_connector_overview.md#Lx-Ly`

- **Projection**  
  Vision hidden → text hidden (Docling often 576).  
  Ref: `foundation-model-stack/fms/models/idefics3/vision_connector_overview.md#Lx-Ly`

- **Packing semantics**  
  Placeholders match `T_img`; derive from encoder output.  
  Ref: `foundation-model-stack/fms/models/idefics3/pack_overview.md#Lx-Ly`

- **Generation**  
  Check `inputs_embeds` path; fallback to `input_ids`.  
  Refs: `foundation-model-stack/fms/models/idefics3/pack_overview.md#Lx-Ly`, `transformers/src/transformers/models/smolvlm/modeling_smolvlm.py#Lx-Ly`

---

## References (TODO: add permalinks + line ranges)

### This repo (FMS design docs)
- Porting overview:  
  `https://github.com/toddllm/smolvlm-fms-design-docs/blob/main/foundation-model-stack/fms/models/idefics3/porting_overview.md#Lx-Ly`
- Vision connector geometry:  
  `https://github.com/toddllm/smolvlm-fms-design-docs/blob/main/foundation-model-stack/fms/models/idefics3/vision_connector_overview.md#Lx-Ly`
- Packing semantics:  
  `https://github.com/toddllm/smolvlm-fms-design-docs/blob/main/foundation-model-stack/fms/models/idefics3/pack_overview.md#Lx-Ly`
- Multi-image / multi-page packing:  
  `https://github.com/toddllm/smolvlm-fms-design-docs/blob/main/foundation-model-stack/fms/models/idefics3/pack_multi_overview.md#Lx-Ly`
- Preprocessing policy:  
  `https://github.com/toddllm/smolvlm-fms-design-docs/blob/main/foundation-model-stack/fms/models/idefics3/preprocessing_overview.md#Lx-Ly`
- HF adapter / mapping:  
  `https://github.com/toddllm/smolvlm-fms-design-docs/blob/main/foundation-model-stack/fms/models/idefics3/hf_adapter_overview.md#Lx-Ly`  
  `https://github.com/toddllm/smolvlm-fms-design-docs/blob/main/foundation-model-stack/fms/models/idefics3/fms_vs_transformers_overview.md#Lx-Ly`

### Upstream (HF / Mistral / vLLM)
- HF SmolVLM config:  
  `https://github.com/huggingface/transformers/blob/main/src/transformers/models/smolvlm/configuration_smolvlm.py#Lx-Ly`
- HF SmolVLM modeling:  
  `https://github.com/huggingface/transformers/blob/main/src/transformers/models/smolvlm/modeling_smolvlm.py#Lx-Ly`
- HF SmolVLM image processor:  
  `https://github.com/huggingface/transformers/blob/main/src/transformers/models/smolvlm/image_processing_smolvlm.py#Lx-Ly`
- HF Idefics-3 (connector lineage):  
  `https://github.com/huggingface/transformers/tree/main/src/transformers/models/idefics3`
- HF SigLIP:  
  `https://github.com/huggingface/transformers/tree/main/src/transformers/models/siglip`
- Mistral Pixtral / Small-3.1 Vision:  
  **TODO add model card / docs permalink**
- vLLM multimodal placeholder contract:  
  **TODO add docs permalink**
```
