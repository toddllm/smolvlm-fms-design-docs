# Connector: Idefics3Connector

This document describes the `Idefics3Connector` class in the SmolVLM foundation-model-stack. The connector bridges the vision and text towers by downsampling the patch grid and projecting it into the text hidden dimension.

## How it works

- **Pixel unshuffle:** The method `pixel_unshuffle` takes a batch of patch embeddings of shape `(B, H*W, C_vit)` and reshapes it into a 2-D grid `(H, W)`. It then uses a downsampling factor `scale` to fold neighbouring patches into the channel dimension. For example, with a 32x32 patch grid and scale 4, the grid is partitioned into 4x4 blocks; each block’s 16 patch embeddings are concatenated along the channel dimension. The result is a grid of shape `(H/scale, W/scale)` with each token having dimension `scale^2 * C_vit`. This operation asserts that `H` and `W` are divisible by `scale`.

- **Linear projection:** The connector then applies a bias-free linear projection that maps from `scale^2 * C_vit` to the text hidden size (`D_text`). The number of input features is computed as `vision_hidden * scale^2` and the number of output features is `text_hidden`. For SmolVLM, `vision_hidden` is 768 and `text_hidden` is 576, and the default `scale` is 4, so the projection maps 12,288-dimensional vectors to 576-dimensional vectors.

- **Forward pass:** In the forward method, the connector first calls `pixel_unshuffle` on the patch embeddings to obtain a tensor of shape `(B, (H/scale)*(W/scale), scale^2*C_vit)`. It checks that the last dimension matches the expected number of input features. It then applies the linear projection to produce an output tensor of shape `(B, (H/scale)*(W/scale), D_text)`.

## Design considerations

- **Downsampling factor:** The default scale of 4 is chosen for SmolVLM so that a 32x32 patch grid is reduced to an 8x8 grid. Models using different vision towers (e.g. NaViT for Idefics3-8B) may use a different scale (2 for a 26x26 grid). The connector’s `scale` parameter should be read from the model configuration rather than hard coded.

- **Bias-free projection:** The projection layer omits a bias term because the concatenation of multiple patch embeddings already captures a rich set of features. Removing the bias slightly reduces the parameter count without affecting model capacity.

- **Static helper:** The `pixel_unshuffle` function is implemented as a static method because it does not depend on any instance fields except the scale. It uses PyTorch’s `unfold` operation to extract local blocks efficiently.

This connector is one of the key building blocks of SmolVLM. In the next documentation file we will cover the packer that inserts the visual tokens into the text sequence.
