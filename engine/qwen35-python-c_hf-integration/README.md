# Qwen3.5 Python/C++ HF Integration

This directory contains the Python and C++ implementation for integrating HoloQubed with Hugging Face Qwen3.5 models.

## Files
* `geometry_forge.py`: Transforms dense Flatland AI models into Complex Hilbert Phase Space for Qwen3.5. Similar to other implementations, it maps dense arrays into 1D continuous spatial distances using JIT-compiled Hilbert curves, implementing OOM protection via connection throttling to manage worker memory consumption during heavy generation tasks.
* `holo_ext.cpp`: Native OpenCL C++ Extension for Qwen3.5 models. A multi-GPU JIT Rehydrator that executes a phase-scaled SpMV OpenCL kernel (`spmv_hilbert_quantum_batched`) to reconstruct amplitudes from decoupled real and imaginary weights safely utilizing global atomic limits.
* `holo_generate_ext.py`: Text generation engine using OpenCL extension for Qwen3.5. Monkey-patches Hugging Face architectures.
* `holo_loader.py`: Ingress loader and CPU Query Planner.
* `holo_setup.py`: Standard setup script for building and installing the C++ PyTorch extension using `torch.utils.cpp_extension`.
