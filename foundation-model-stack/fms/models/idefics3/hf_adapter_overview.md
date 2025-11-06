# HF Adapter Overview  

## Why an HF Adapter?  
- The Foundation Model Stack re-implements models to leverage PyTorch-native features (torch.compile, FSDP, etc.) for optimized inference and training ([github.com](https://github.com/foundation-model-stack/foundation-model-stack#:~:text=Foundation%20Model%20Stack%20Foundation%20Model,To%20enable)).  
- To maintain compatibility with the Hugging Face Transformers ecosystem, FMS provides adapter APIs such as `to_hf_api`. This allows converting an FMS model into a Hugging Face‑style model so that you can call `.generate` or use the model with HF pipelines.  
- This page explains how the adapter works and how it maps FMS components onto the HF interface.  

## Adapter patterns in FMS  
- Each FMS model defines a `to_hf_api` method, which returns an object conforming to the HF `PreTrainedModel` interface.  
- The adapter wraps the underlying text model and registers pre-hooks so that image processing is still handled at the FMS level before the HF forward call.  
- For example, the Idefics3 FMS model's `to_hf_api` method exposes `generate` and `forward` methods that call the internal `prepare_inputs` pre-hook to splice image embeddings, then delegate to the text backbone's generation method.  

## Example: SmolVLM adapter  
- In the FMS `model.py` for Idefics3, the `to_hf_api` method returns an instance of `HFModelAdapter` with references to the internal modules.  
- When you call `.generate` on the adapter, it will internally call `prepare_inputs` to process images and then call the text model's `generate`, aligning with HF semantics.  
- This pattern allows you to use FMS models within HF pipelines without rewriting code.
