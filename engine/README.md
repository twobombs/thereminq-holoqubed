# ThereminQ Holoqubed Engine

This directory contains the core execution scripts and utilities for the ThereminQ Holoqubed AI inference engine.

<img width="2752" height="1536" alt="unnamed (4)" src="https://github.com/user-attachments/assets/8adbf464-e9d2-49a7-a20b-d059af25f6d3" />

---

## Overview

ThereminQ Holoqubed is a sparse, holographic AI inference engine that bypasses dense matrix math via O(1) spatial queries and FP16 Rapid Packed Math. It encodes floating-point activation thresholds into 1D spatial coordinates using a custom bit-interleaving scheme (bitwise XOR) forming a Z-order (Morton) curve, avoiding traditional Hilbert curve algorithms. The engine simulates a large System RAM dictionary using Zero-Copy bindings and executes custom OpenCL kernels to evaluate the Tesseract KV Cache via a Scatter-Gather pattern and hardware SiLU activation.

---

## Subdirectories

### [`concept/`](concept/README.md)

Contains conceptual prototypes and verification harnesses for the Holoqubed engine.

| File | Functionality | Use Cases |
|------|---------------|-----------|
| [`gguf_vs_holo_divergences.py`](concept/gguf_vs_holo_divergences.py) | **Accuracy Harness**: Runs a dense GGUF reference model side-by-side with the sparse `.holo` engine to measure mathematical divergence caused by the SiLU sparsity threshold. | Tune the pruning threshold during the offline forge phase by computing Top-1 Accuracy Match and Mean Squared Error (MSE). |
| [`holo_generate_hf.py`](concept/holo_generate_hf.py) | **Text Generation Engine**: Monkey-patches a Hugging Face `PreTrainedModel` by replacing dense `nn.Linear` layers with custom `HoloLinear` PyTorch modules backed by an OpenCL SpMV engine. | Dynamic forward pass interception for `AutoModelForCausalLM` and `AutoTokenizer` execution. |
| [`holoqubed_prototype.py`](concept/holoqubed_prototype.py) | **PyOpenCL Prototype**: A sparse, holographic AI inference engine targeting AMD Radeon Pro V340 (Vega 10) hardware via Mesa Rusticl. | Simulate 320GB System RAM dictionary using Zero-Copy bindings and evaluate Tesseract KV Cache via Scatter-Gather pattern. |

---

### [`gpt2-python-c_hf-implementation/`](gpt2-python-c_hf-implementation/README.md)

Python and C++ implementation for integrating Holoqubed with Hugging Face GPT-2 models.

| File | Functionality | Use Cases |
|------|---------------|-----------|
| [`geometry_forge.py`](gpt2-python-c_hf-implementation/geometry_forge.py) | Transforms dense Flatland AI models into Complex Hilbert Phase Space for GPT-2. Maps 2D Cartesian coordinates into a 1D continuous spatial distance using a JIT-compiled Hilbert Curve engine. | Inject deterministic spatial phase shift, casting real-valued magnitudes into Complex Cartesian space (Real + Imaginary) for phase modulation during SpMV execution. |
| [`holo_ext.cpp`](gpt2-python-c_hf-implementation/holo_ext.cpp) | **Native OpenCL C++ Extension**: Multi-GPU JIT Rehydrator that loads sparse quantum weights, reconstructs original signed amplitudes, modulates them using spatial frequency interference, and outputs a dense matrix representation. | Bypass atomic operations entirely via static pre-allocation. |
| [`holo_generate_ext.py`](gpt2-python-c_hf-implementation/holo_generate_ext.py) | **Text Generation Engine**: Monkey-patches Hugging Face `PreTrainedModel` replacing dense linear layers with custom OpenCL-backed `HoloLinear` implementations. | GPT-2 text generation with OpenCL acceleration. |
| [`holo_loader.py`](gpt2-python-c_hf-implementation/holo_loader.py) | **Ingress Loader & CPU Query Planner**: Exposes `HoloQueryPlanner` which mounts the compressed `.holo` dictionary directly from disk using `zipfile`. | Fast layer loading and pathway lookup. |
| [`holo_setup.py`](gpt2-python-c_hf-implementation/holo_setup.py) | Standard setup script for building and installing the C++ PyTorch extension using `torch.utils.cpp_extension`. | Extension installation and environment preparation. |

---

### [`hilberspace-python-c_hf-implementation/`](hilberspace-python-c_hf-implementation/README.md)

Implementation for Hilberspace with Python and C++ integration for Hugging Face models.

| File | Functionality | Use Cases |
|------|---------------|-----------|
| [`geometry_forge.py`](hilberspace-python-c_hf-implementation/geometry_forge.py) | Transforms dense Flatland AI models into Complex Hilbert Phase Space. Pre-computes Hilbert curve topologies and Cartesian Phase (Real/Imaginary) for pure FP32 zero-branching OpenCL execution. | Dynamic scalpel leveraging both standard deviation and percentile thresholds to safely prune weights. |
| [`holo_ext.cpp`](hilberspace-python-c_hf-implementation/holo_ext.cpp) | **Native OpenCL C++ Extension**: Provides a highly optimized `spmv_hilbert_quantum_batched` kernel implementing stable phase-scaling architecture. Loads explicitly decoupled Real and Imaginary weights, reconstructs original amplitudes, and scales them by passed phase cosines rather than computing them per-thread. | Utilizes OpenCL atomics to safely reduce outputs in global memory. |
| [`holo_generate_ext.py`](hilberspace-python-c_hf-implementation/holo_generate_ext.py) | **Text Generation Engine**: Modifies the Hugging Face generation loop to inject the `HoloLinear` layer, passing in phase cosines dynamically during the batched execution pass. | Batched Hilberspace inference with dynamic phase modulation. |
| [`holo_loader.py`](hilberspace-python-c_hf-implementation/holo_loader.py) | **Ingress Loader & CPU Query Planner**: Reads and unzips the spatially-ordered payloads created by `geometry_forge.py`. | Fast payload extraction and layer mounting. |
| [`holo_setup.py`](hilberspace-python-c_hf-implementation/holo_setup.py) | Standard setup script for building and installing the C++ PyTorch extension using `torch.utils.cpp_extension`. | Extension installation and environment preparation. |

---

### [`multithreaded-python-c_hf-implementation/`](multithreaded-python-c_hf-implementation/README.md)

Multithreaded Python and C++ implementation for integrating Holoqubed with Hugging Face models.

| File | Functionality | Use Cases |
|------|---------------|-----------|
| [`holo_ext.cpp`](multithreaded-python-c_hf-implementation/holo_ext.cpp) | **Native OpenCL C++ Extension**: Resolves concurrency issues by dynamically allocating Input/Output OpenCL buffers per forward pass within the `NativeHoloLayer` rather than statically pinning them. | Ensures thread-safety while preserving persistent pinning of static coordinates and weights in VRAM. |
| [`holo_generate_ext.py`](multithreaded-python-c_hf-implementation/holo_generate_ext.py) | **Text Generation Engine**: Supports concurrent generation threads using Python's `concurrent.futures.ThreadPoolExecutor` combined with `threading.Lock()`. | Spin up multiple parallel prompt executions with isolated PyTorch/C++ context switches across threads. |
| [`holo_loader.py`](multithreaded-python-c_hf-implementation/holo_loader.py) | **Ingress Loader & CPU Query Planner**: Fast layer loading and pathway lookup for multithreaded execution. | Concurrent model loading and query planning. |
| [`holo_setup.py`](multithreaded-python-c_hf-implementation/holo_setup.py) | Standard setup script for building and installing the C++ PyTorch extension using `torch.utils.cpp_extension`. | Extension installation and environment preparation. |

---

### [`python-c_hf-implementation/`](python-c_hf-implementation/README.md)

Core Python and C++ implementation for integrating Holoqubed with Hugging Face models. Works with GGUF and D34 models.

| File | Functionality | Use Cases |
|------|---------------|-----------|
| [`gguf2holo.py`](python-c_hf-implementation/gguf2holo.py) | **Offline Model Ingress & Conversion Pipeline**: Converts dense `.gguf` and `.pt` models (FP32/FP16/BF16) into the sparse, spatially encoded `.holo` dictionary. Implements Hybrid Dynamic Scalpel using distribution-blind (standard deviation) and distribution-aware (percentile) thresholds. | Batched Morton (Z-order) bit-interleaving via `np.bitwise_xor.reduce` for dynamic 1D spatial encoding. Parallel tensor processing with OOM protection and Zstandard dictionary injection for optimized compression. |
| [`holo_loader.py`](python-c_hf-implementation/holo_loader.py) | **Ingress Loader & JIT CPU Query Planner**: Exposes `HoloQueryPlanner` which mounts the compressed `.holo` dictionary directly from disk using `zipfile`. Extracts embedded Zstandard dictionary to rapidly decompress payload layers on the fly. | Ultra-fast O(log N) pathway lookups via `np.searchsorted` to extract only active spatial coordinates and weights. |
| [`holo_ext.cpp`](python-c_hf-implementation/holo_ext.cpp) | **Native OpenCL C++ Extension**: Provides pre-inflated native OpenCL execution kernels for scattering sparse weights and performing SpMV dense math. Pins decompressed `coords` and `weights` to persistent VRAM buffers. | Dynamic OpenCL local memory (`__local` cache) management based on output feature size. Executes Grid-Stride Local Reduction SpMV kernel (`spmv_holo_weights_fast`) using OpenCL atomics. Integrates via `pybind11`. |
| [`holo_generate_ext.py`](python-c_hf-implementation/holo_generate_ext.py) | **Pure OpenCL Text Generation Edition**: Monkey-patches Hugging Face `PreTrainedModel` by replacing dense linear layers with custom `HoloLinear` module. | Scatters sparse Morton coordinates into persistent OpenCL VRAM buffers during initialization. Execution happens purely via OpenCL PyTorch C++ extension, bypassing CPU overhead. |
| [`holo_check.py`](python-c_hf-implementation/holo_check.py) | **Ingress Loader & Simulator**: Simulates CPU Query Planner loading layers from `.holo` dictionary and executes direct OpenCL SpMV in VRAM using built-in OpenCL compilation chain via `pyopencl`. | Validates OpenCL functionality and simulates memory utilization savings. |
| [`holo_setup.py`](python-c_hf-implementation/holo_setup.py) | Standard setup script for building and installing the `holo_ext` C++ PyTorch extension using `torch.utils.cpp_extension`. | Extension installation and environment preparation. |
| [`holo_setup.sh`](python-c_hf-implementation/holo_setup.sh) | Shell script wrapper to automate execution of `holo_setup.py` and environment preparation. | Quick setup and environment configuration. |

---

### [`qwen35-python-c_hf-integration/`](qwen35-python-c_hf-integration/README.md)

Python and C++ implementation for integrating Holoqubed with Hugging Face Qwen3.5 models.

| File | Functionality | Use Cases |
|------|---------------|-----------|
| [`geometry_forge.py`](qwen35-python-c_hf-integration/geometry_forge.py) | Transforms dense Flatland AI models into Complex Hilbert Phase Space for Qwen3.5. Maps dense arrays into 1D continuous spatial distances using JIT-compiled Hilbert curves. | OOM protection via connection throttling to manage worker memory consumption during heavy generation tasks. |
| [`holo_ext.cpp`](qwen35-python-c_hf-integration/holo_ext.cpp) | **Native OpenCL C++ Extension**: Multi-GPU JIT Rehydrator that executes phase-scaled SpMV OpenCL kernel (`spmv_hilbert_quantum_batched`) to reconstruct amplitudes from decoupled real and imaginary weights safely utilizing global atomic limits. | Multi-GPU inference with phase-scaled reconstruction. |
| [`holo_generate_ext.py`](qwen35-python-c_hf-integration/holo_generate_ext.py) | **Text Generation Engine**: Monkey-patches Hugging Face architectures for Qwen3.5 models. | Qwen3.5 text generation with OpenCL acceleration. |
| [`holo_loader.py`](qwen35-python-c_hf-integration/holo_loader.py) | **Ingress Loader & CPU Query Planner**: Fast layer loading and pathway lookup. | Efficient model loading for Qwen3.5. |
| [`holo_setup.py`](qwen35-python-c_hf-integration/holo_setup.py) | Standard setup script for building and installing the C++ PyTorch extension using `torch.utils.cpp_extension`. | Extension installation and environment preparation. |

---

## Quick Start

1. **Convert a model**: Run `gguf2holo.py` to convert a dense `.gguf` or `.pt` model into a sparse `.holo` dictionary.
2. **Generate text**: Use `holo_generate_ext.py` to run inference with the converted model.
3. **Validate**: Use `holo_check.py` to simulate and validate OpenCL functionality.

---

## Architecture

The Holoqubed engine operates on the following principles:

1. **Spatial Encoding**: Dense weights are transformed into complex Hilbert phase space using Morton (Z-order) curves.
2. **Sparse Pruning**: A hybrid dynamic scalpel uses both standard deviation and percentile thresholds to safely prune weights.
3. **OpenCL Acceleration**: Native C++ extensions provide highly optimized OpenCL kernels for SpMV operations.
4. **Zero-Copy Memory**: Persistent VRAM buffers enable fast data transfer between CPU and GPU.
5. **JIT Rehydration**: Sparse weights are reconstructed on-the-fly during inference using phase-scaled SpMV.
