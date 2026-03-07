# ThereminQ Holoqubed 🌌

**An experimental, quantum-inspired, sparse holographic AI inference engine.**

ThereminQ Holoqubed is a radical departure from traditional dense neural network execution. Instead of relying on brute-force, dense matrix multiplications ($O(n^2)$) that bottleneck on compute cores and PCIe bandwidth, Holoqubed leverages high-speed memory bandwidth, $O(1)$ spatial coordinate lookups, and bit-interleaved geometric encoding to perform AI inference.

Currently in the prototyping phase, the engine is being modeled in Python (`PyOpenCL` + `Weed`) 

---

## Core Architecture

Traditional Large Language Models (LLMs) push massive weight matrices across the PCIe bus for every single token. Holoqubed bypasses this by translating neural pathways into physical memory space:

* **The Holographic Dictionary:** Stored in massive system RAM (e.g., 320GB). It maps token pathways as spatial coordinates rather than dense weights.
* **Hilbert Curve Encoding:** Converts floating-point activation thresholds into 1D spatial coordinates using bitwise XOR interleaving (e.g., `index ^= (val << (i * 2))`).
* **Tesseract KV Cache:** Represents a 4D coordinate space mapping the active sequence generation.
* **Sparse Execution:** Resolves $O(1)$ lookups on the CPU and only pushes active coordinate pathways across the PCIe bus to the GPU. This effectively neutralizes traditional PCIe bottlenecks, allowing the engine to run over x4 connections.

---

## 💻 Hardware Target

This engine is being co-designed alongside a specific enterprise hardware topology optimized for memory bandwidth and Rapid Packed Math (FP16):

* **Motherboard:** Supermicro Dual-EPYC (e.g., H11DSi)
* **System RAM:** 320GB+ (acting as the Holographic Dictionary pool)
* **GPUs:** 3x AMD Radeon Pro V340 (Dual Vega 10 / `gfx900` dies)
* **Connection:** PCIe 3.0 x4 

---

## 🐳 Software Stack: Containerized Rusticl

Holoqubed does not use AMD's proprietary drivers. Instead, it runs inside a **Ubuntu 26.04 Docker container** leveraging **Mesa's Rusticl**, a modern, Rust-based OpenCL 3.0 implementation that fully supports `radeonsi` and Rapid Packed Math.

### 1. Building the Engine
You can build the Docker container locally using the provided `Dockerfile`. This pulls the latest Mesa drivers, Python, and PyOpenCL.

```bash
docker build -t holoqubed:latest .
