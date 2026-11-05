# Multithreaded Python/C++ HF Implementation

This directory contains a multithreaded Python and C++ implementation for integrating HoloQubed with Hugging Face models.

## Files
* `holo_ext.cpp`: Native OpenCL C++ Extension. Resolves concurrency issues by dynamically allocating Input/Output OpenCL buffers per forward pass within the `NativeHoloLayer` rather than statically pinning them. This ensures thread-safety while preserving the persistent pinning of the static coordinates and weights in VRAM.
* `holo_generate_ext.py`: Text generation engine using OpenCL extension, supporting concurrent generation threads. Utilizes Python's `concurrent.futures.ThreadPoolExecutor` combined with `threading.Lock()` to spin up multiple parallel prompt executions. It safely isolates the PyTorch/C++ context switches across threads.
* `holo_loader.py`: Ingress loader and CPU Query Planner.
* `holo_setup.py`: Standard setup script for building and installing the C++ PyTorch extension using `torch.utils.cpp_extension`.
