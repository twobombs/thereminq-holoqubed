# Old Llama Integration Scripts

This directory contains older, experimental scripts for patching and integrating `llama.cpp` with the HoloQubed sparse inference engine logic.

## Files

*   `fix_ggml_size.py`: Script to patch ggml sizes.
*   `llama_holo_intercept.py`: Script intercepting `llama.cpp` math execution for holo integration.
*   `llama_holo_patch.py`: Original patching script for llama.
*   `rewrite-ggml-holo-patch-4threads.py`: Multi-threaded patch script for ggml.