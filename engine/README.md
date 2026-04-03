# ThereminQ Holoqubed Engine

This directory contains the core execution scripts and utilities for the ThereminQ Holoqubed AI inference engine.

<img width="2752" height="1536" alt="unnamed (4)" src="https://github.com/user-attachments/assets/8adbf464-e9d2-49a7-a20b-d059af25f6d3" />


## Subdirectories

*   **`concept/`**: Contains conceptual prototypes and verification harnesses (e.g., prototype PyOpenCL loom, Hugging Face monkey-patching). See `concept/README.md`.
*   **`python-c_hf-implementation/`**: Contains the Python and C++ implementation for integrating HoloQubed with Hugging Face models (converter, C++ OpenCL extension, etc.). See `python-c_hf-implementation/README.md`.

## Files

*   `holo_check.py`: Ingress Loader & Simulator. It acts as an OpenCL SpMV simulator and validation tool. Specifically, it simulates the CPU Query Planner loading layers from a compressed `.holo` dictionary (via `zstandard` reading from a ZIP archive) and executes direct OpenCL Sparse Matrix-Vector Multiplication (SpMV) in VRAM. It validates hardware detection via `pyopencl`, performs ultra-fast O(log N) pathway lookups, and mocks the execution of a spatial cloud of active coordinates to verify mathematical execution correctness and memory requirements prior to full pipeline integration.
