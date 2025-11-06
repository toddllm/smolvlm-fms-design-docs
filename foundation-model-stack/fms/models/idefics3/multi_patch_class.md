# MultiPatchVisionProcessor Class

This page will describe the `MultiPatchVisionProcessor` class.

## Overview

- Splits large images into multiple 512×512 patches.
- Processes each patch with the SigLIP vision encoder and a connector.
- Concatenates the resulting visual tokens to form a single sequence.

## Next steps

In subsequent pages we will explain the class purpose, attributes, and processing logic in detail, with references to the SmolVLM configuration and original implementation.
