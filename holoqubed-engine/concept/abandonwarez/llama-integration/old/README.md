# Old Llama Integration Scripts

This directory contains older, experimental scripts for patching and integrating `llama.cpp` with the HoloQubed sparse inference engine logic.

## Files

*   `fix_ggml_size.py`: Script to patch ggml sizes. It forcibly overrides the `ggml` memory allocator via string replacement in `ggml.c` to reserve 8 bytes per element for the custom `HOLO_SPARSE` type structure.
*   `llama_holo_intercept.py`: Script intercepting `llama.cpp` math execution for holo integration. It physically writes a new `ggml-holo.c` file containing the OpenCL context generation logic into the `llama.cpp` source tree, then rewrites `ggml-cpu.c` to `#include` it and trigger execution when the `GGML_TYPE_HOLO_SPARSE` type is encountered during matrix multiplication.
*   `llama_holo_patch.py`: Original patching script for llama. It hooks directly into the GGML type enum structures in `ggml.h` and `ggml.c` to officially register the `GGML_TYPE_HOLO_SPARSE` type alongside existing standard quantizations.
*   `rewrite-ggml-holo-patch-4threads.py`: Multi-threaded patch script for ggml. An iterative upgrade over `llama_holo_intercept.py` that utilizes `#pragma omp critical` to create a thread-safe mutex, preventing collision during OpenCL kernel initialization across the 48-thread environment of modern CPUs running `llama.cpp`.
