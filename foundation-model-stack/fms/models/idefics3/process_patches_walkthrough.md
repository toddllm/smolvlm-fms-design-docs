# process_patches Walkthrough

This page will provide a step-by-step explanation of the `process_patches` method in the `MultiPatchVisionProcessor` class. We'll explain what each part of the function does, referencing the original implementation in the FMS repository and the relevant configuration values.

## Outline

- Loop over patch tensors to extract features using the vision encoder.
- Validate the shape of the output tokens and ensure they match the expected `(tokens_per_patch, hidden_dim)` size.
- Optionally downsample using the connector and check that the output length equals `tokens_per_patch`.
- Concatenate the resulting embeddings from all patches into a single tensor.
- Return the concatenated embeddings and a stats dictionary.

We'll elaborate on each of these steps in future iterations and link to specific code lines in `multi_patch.py` once we flesh out the details.
