# ThereminQ Holoqubed Engine

This directory contains the core execution scripts and utilities for the ThereminQ Holoqubed AI inference engine.

## Subdirectories

*   **`concept/`**: Contains conceptual prototypes and verification harnesses (e.g., prototype PyOpenCL loom, Hugging Face monkey-patching). See `concept/README.md`.
*   **`python-c_hf-implementation/`**: Contains the Python and C++ implementation for integrating HoloQubed with Hugging Face models (converter, C++ OpenCL extension, etc.). See `python-c_hf-implementation/README.md`.

## Files

*   `holo_check.py`: Ingress Loader & Simulator. Simulates the CPU Query Planner loading layers from a `.holo` dictionary and executes a direct OpenCL Sparse Matrix-Vector Multiplication (SpMV) in VRAM to verify the OpenCL execution pipeline.
