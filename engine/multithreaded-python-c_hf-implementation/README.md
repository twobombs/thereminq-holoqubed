# Multithreaded Python/C++ HF Implementation

This directory contains a multithreaded Python and C++ implementation for integrating HoloQubed with Hugging Face models.

## Files
* `holo_ext.cpp`: Native OpenCL C++ Extension.
* `holo_generate_ext.py`: Text generation engine using OpenCL extension, supporting concurrent generation threads.
* `holo_loader.py`: Ingress loader and CPU Query Planner.
* `holo_setup.py`: Standard setup script for building and installing the C++ PyTorch extension.
