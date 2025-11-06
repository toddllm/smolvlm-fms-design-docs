# Patterns in FMS vs Transformers

This page outlines general patterns we observe when porting models from Hugging Face's Transformers into the Foundation Model Stack (FMS). It serves as a reference for how the FMS repository organizes model implementations and how these patterns compare to the structure of Transformers models.

## FMS design principles

- **Optimized inference and training**: FMS reimplements popular architectures (e.g., LLaMA, GPT‑BigCode) to leverage PyTorch-native features like `torch.compile`, scaled dot-product attention, and FSDP for efficient distributed training ([github.com](https://github.com/foundation-model-stack/foundation-model-stack#:~:text=Foundation%20Model%20Stack%20Foundation%20Model,To%20enable)).
- **Modular components**: Models are broken down into vision towers, connectors, packers, and text backbones, mirroring the modularity seen in Transformers but with FMS-specific optimizations.
- **Adapter APIs**: FMS provides methods such as `to_hf_api` to convert an FMS model into a Hugging Face compatible one, preserving API parity while benefiting from the FMS optimizations.

## Transformers structure

- In the Transformers repo, each model has a configuration file, a modeling file, and optional processing modules. For example, SmolVLM includes:
  - `configuration_smolvlm.py` for configuration definitions.
  - `modeling_smolvlm.py` for forward pass and generation.
  - `image_processing_smolvlm.py` for preprocessing images ([github.com](https://github.com/huggingface/transformers/tree/main/src/transformers/models/smolvlm)).

These patterns inform how we organize our documentation and porting efforts.
