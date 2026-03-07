# ThereminQ HoloQubed 🌌

**An experimental, quantum-inspired, sparse holographic AI inference engine.**

ThereminQ HoloQubed is a radical departure from traditional dense neural network execution. Instead of relying on brute-force, dense matrix multiplications ($O(n^2)$) that bottleneck on compute cores and PCIe bandwidth, Holoqubed leverages high-speed memory bandwidth, $O(1)$ spatial coordinate lookups, and bit-interleaved geometric encoding to perform AI inference.

Currently in the prototyping phase, the engine is built in Python (`PyOpenCL` + `Weed`) to map the mathematical abstractions before being ported to bare-metal C++ for absolute maximum PCIe Zero-Copy efficiency.

---

## Core Architecture

Traditional Large Language Models (LLMs) push massive weight matrices across the PCIe bus for every single token. Holoqubed bypasses this by translating neural pathways into physical memory space:

* **The Holographic Dictionary:** Stored in massive system RAM (e.g., 320GB). It maps token pathways as spatial coordinates rather than dense weights.
* **Hilbert Curve Encoding:** Converts floating-point activation thresholds into 1D spatial coordinates using bitwise XOR interleaving.
* **Tesseract KV Cache:** Represents a 4D coordinate space mapping the active sequence generation.
* **Sparse Execution:** Resolves $O(\log N)$ or $O(1)$ lookups on the CPU and only pushes active coordinate pathways across the PCIe bus to the GPU. This effectively neutralizes traditional PCIe bottlenecks, allowing the engine to run over x4 connections.

---

## 🛠️ The Holoqubed Toolchain

The repository contains a complete pipeline to convert, load, execute, and verify holographic models.

### 1. The Offline Forge (`gguf2holo.py`)

Converts standard dense `.gguf` models into the highly optimized, memory-mappable `.holo` format. It applies the "Holoqubed Collapse" (threshold pruning) to eliminate mathematically insignificant weights and encodes the surviving pathways into 1D spatial coordinates.

### 2. The CPU Query Planner (`holo-loader.py`)
Memory-maps (`mmap`) the massive `.holo` dictionary to disk, allowing the system RAM to act as a zero-latency cache. When the engine generates spatial coordinates, the Query Planner performs sub-millisecond $O(\log N)$ binary searches to extract only the necessary FP16 pathways to send to the GPUs.

### 3. The Multi-GPU Loom (`holoqubed_prototype.py`)
The PyOpenCL execution engine. It automatically detects all available Vega 10 dies, establishes a unified Zero-Copy memory bridge, and uses a **Scatter-Gather** pattern to distribute spatial lookups across the hardware. It executes a custom Rapid Packed Math (`half2`) kernel for hardware-accelerated SiLU activation and Top-K filtering.

### 4. The Accuracy Harness (`gguf_vs_holo_divergence.py`)

Runs a dense `llama.cpp` reference model side-by-side with the sparse `.holo` engine to measure mathematical divergence. This is used to tune the sparsity threshold during the offline forge to ensure the engine retains maximum intelligence while dropping dead weight.

---

## 💻 Hardware Target

This engine is being co-designed alongside a specific enterprise hardware topology optimized for memory bandwidth and Rapid Packed Math (FP16):

* **Motherboard:** Supermicro Dual-EPYC (e.g., H11DSi)
* **System RAM:** 320GB+ (acting as the Holographic Dictionary pool)
* **GPUs:** 3x AMD Radeon Pro V340 (6x Vega 10 / `gfx900` dies) + NVidia/Intel OpenCL support
* **Connection:** PCIe 3.0 x4 via shielded risers.

---

## 🐳 Software Stack: Containerized Rusticl

Holoqubed does not use AMD's proprietary drivers. Instead, it runs inside a lightweight **Ubuntu 26.04 Docker container** leveraging **Mesa's Rusticl**, a modern, Rust-based OpenCL 3.0 implementation that fully supports `radeonsi` and Rapid Packed Math.

### Building the Environment
You can build the Docker container locally using the provided `Dockerfile`.

```bash
docker build -t holoqubed:latest .
