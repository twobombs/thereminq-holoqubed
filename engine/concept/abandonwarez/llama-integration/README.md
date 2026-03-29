<img width="2816" height="1536" alt="Gemini_Generated_Image_jsrig8jsrig8jsri" src="https://github.com/user-attachments/assets/a0313af8-5414-4d9e-9b81-1014f7df0376" />

# Llama Integration Scripts

This directory contains older, experimental scripts for patching and integrating `llama.cpp` with the HoloQubed sparse inference engine logic.

## Files

*   `hook_ggml_core.py`: Script to safely patch and interact with ggml core components.
*   `pack_llama_holo.py`: Logic for packing llama execution states.
*   `patch-llama.sh`: Shell script to apply patching.