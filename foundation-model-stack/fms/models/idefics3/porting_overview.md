# Porting SmolVLM from HuggingFace to Foundation Model Stack

This page provides a high-level overview of how the SmolVLM implementation in Hugging Face’s Transformers library is adapted within the Foundation Model Stack (FMS). Understanding the motivations behind FMS and the differences between the two implementations will help contextualize the design decisions in this documentation.

## Why reimplement in FMS?

- The Foundation Model Stack aims to provide optimized inference and training for large models using PyTorch-native features such as `torch.compile`, accelerated transformers (scaled dot product attention), and Fully Sharded Data Parallel (FSDP). The FMS README describes reimplementations of popular architectures (e.g., LLaMA and GPT-BigCode) to enable these optimizations ([github.com](https://github.com/foundation-model-stack/foundation-model-stack#:~:text=Foundation%20Model%20Stack%20Foundation%20Model,To%20enable)).
- By porting SmolVLM into FMS, we can leverage these optimizations while maintaining compatibility with Hugging Face APIs via adapters.

## Hugging Face SmolVLM structure

- In the Transformers repo, the SmolVLM model is defined under `src/transformers/models/smolvlm/` with modules such as:
  - `configuration_smolvlm.py` for the model configuration.
  - `modeling_smolvlm.py` for the model’s forward pass and generation logic.
  - `image_processing_smolvlm.py` and related utilities for preprocessing images and videos.
- These modules serve as the reference “ground truth” for behavior and parameter names ([github.com](https://github.com/huggingface/transformers/tree/main/src/transformers/models/smolvlm)).

## Porting patterns

When porting to FMS, several patterns emerge:

- **Shared configuration**: FMS uses the same hyperparameters (e.g., `hidden_size`, `resampler_n_latents`, `pixel_shuffle_factor`) as the Hugging Face config. Our documentation always cites these values from the original config to anchor the design.
- **Vision and connector reuse**: The FMS implementation reuses the SigLIP vision encoder and pixel-unshuffle connector semantics but reimplements them using FMS modules to facilitate `torch.compile` and FSDP.
- **Adapter compatibility**: The FMS code provides adapters to convert FMS models back into Hugging Face-compatible models (`to_hf_api`), enabling interoperability with Transformers pipelines, as noted in the FMS README ([github.com](https://github.com/foundation-model-stack/foundation-model-stack#:~:text=see%20here%20HF%20Model%20Support,A)).
- **Extensible processing**: FMS introduces multi-patch processing for high-resolution images while maintaining the same token counts as the original SmolVLM by adhering to the `resampler_n_latents` value from the config.

In upcoming pages, we will dive into these patterns with concrete code excerpts and comparisons between the FMS and Transformers implementations.
