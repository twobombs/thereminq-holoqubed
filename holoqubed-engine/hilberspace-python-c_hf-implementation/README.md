# Hilberspace Python/C++ HF Implementation

This directory contains the implementation for Hilberspace with Python and C++ integration for Hugging Face models.

## Files
* `geometry_forge.py`: Transforms dense Flatland AI models into Complex Hilbert Phase Space. Similar to the GPT-2 and Qwen3.5 implementations, it pre-computes Hilbert curve topologies and Cartesian Phase (Real/Imaginary) for pure FP32 zero-branching OpenCL execution. It implements a dynamic scalpel leveraging both standard deviation and percentile thresholds to safely prune weights.
* `holo_ext.cpp`: Native OpenCL C++ Extension. Provides a highly optimized `spmv_hilbert_quantum_batched` kernel that implements a stable phase-scaling architecture. It loads explicitly decoupled Real and Imaginary weights, reconstructs original amplitudes, and scales them by passed phase cosines rather than computing them per-thread, utilizing OpenCL atomics to safely reduce outputs in global memory.
* `holo_generate_ext.py`: Text generation engine using OpenCL extension. Modifies the Hugging Face generation loop to inject the `HoloLinear` layer, passing in phase cosines dynamically during the batched execution pass.
* `holo_loader.py`: Ingress loader and CPU Query Planner. Reads and unzips the spatially-ordered payloads created by `geometry_forge.py`.
* `holo_setup.py`: Standard setup script for building and installing the C++ PyTorch extension using `torch.utils.cpp_extension`.

<img width="2000" height="992" alt="Xh5t8" src="https://github.com/user-attachments/assets/8be58f30-99b3-4fbf-9f9b-89235e9c7fe8" />
<img width="1927" height="2000" alt="IcgXt (1)" src="https://github.com/user-attachments/assets/b9916e6b-a0b0-4603-b1dd-722ab55bf6ca" />
