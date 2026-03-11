<img width="1016" height="443" alt="Screenshot from 2026-01-01 15-36-30" src="https://github.com/user-attachments/assets/b5c8d613-5dac-484a-baef-0032dfd8e484" />

# ThereminQ HoloQubed 🌌

**An experimental, quantum-inspired, sparse holographic AI inference engine.**

ThereminQ HoloQubed is a radical departure from traditional dense neural network execution. Instead of relying on brute-force, dense matrix multiplications ($O(n^2)$) that bottleneck on compute cores and PCIe bandwidth, Holoqubed leverages high-speed memory bandwidth, $O(1)$ spatial coordinate lookups, and bit-interleaved geometric encoding to perform AI inference.

Currently in the prototyping phase, the engine is built in Python (`PyOpenCL` + `Weed`) to map the mathematical abstractions before being ported to bare-metal C++ for absolute maximum PCIe Zero-Copy efficiency.

---

## Core Architecture

Traditional Large Language Models (LLMs) push massive weight matrices across the PCIe bus for every single token. HoloQubed bypasses this by translating neural pathways into physical memory space:

* **The Holographic Dictionary:** Stored in massive system RAM (e.g., 320GB). It maps token pathways as spatial coordinates rather than dense weights.
* **Spatial Encoding (Holoqubed Research Logic):** Converts floating-point activation thresholds into 1D spatial coordinates using a custom bit-interleaving scheme (bitwise XOR and shifts), producing a Z-order (Morton) curve.
* **Tesseract KV Cache:** Represents a 4D coordinate space mapping the active sequence generation.
* **Sparse Execution:** Resolves $O(\log N)$ or $O(1)$ lookups on the CPU and only pushes active coordinate pathways across the PCIe bus to the GPU. This effectively neutralizes traditional PCIe bottlenecks, allowing the engine to run over x4 connections.

---

## 🛠️ The HoloQubed Toolchain

The repository contains a complete pipeline to convert, load, execute, and verify holographic models.

### 1. The Offline Forge (`engine/gguf2holo.py`)

Converts standard dense `.gguf` models into the highly optimized, memory-mappable `.holo` format. It uses `np.savez` (uncompressed) instead of compressed formats to enable true zero-copy memory mapping (`mmap_mode='r'`). It applies the "Holoqubed Collapse" (threshold pruning) to eliminate mathematically insignificant weights and encodes the surviving pathways into 1D spatial coordinates.

### 2. The CPU Query Planner (`engine/holo_loader.py`)
Memory-maps (`mmap`) the massive `.holo` dictionary to disk, allowing the system RAM to act as a zero-latency cache. When the engine generates spatial coordinates, the Query Planner performs sub-millisecond $O(\log N)$ binary searches to extract only the necessary FP16 pathways to send to the GPUs.

### 3. The Multi-GPU Loom (`engine/holoqubed_prototype.py`)
The PyOpenCL execution engine. It automatically detects all available Vega 10 dies, establishes a unified Zero-Copy memory bridge, and uses a **Scatter-Gather** pattern to distribute spatial lookups across the hardware. It executes a custom Rapid Packed Math (`half2`) kernel for hardware-accelerated SiLU activation and Top-K filtering.

### 4. The Accuracy Harness (`engine/gguf_vs_holo_divergences.py`)

Runs a dense `llama.cpp` reference model side-by-side with the sparse `.holo` engine to measure mathematical divergence. This is used to tune the sparsity threshold during the offline forge to ensure the engine retains maximum intelligence while dropping dead weight.

---

## 🧰 Tool Cupboard (`tc/`)

A collection of auxiliary utilities and autonomous workflows.

### Deep Local Research (`tc/deep-local-research.py`)

An autonomous research script powered by local LLMs (orchestrator and reasoning models). It performs deep web scraping via DuckDuckGo (filtering out video/image sites), executes a multi-step analysis (Tool Planning, Reasoning, Verification), and outputs a formatted PDF report.

---

## 💻 Hardware Target

This engine is being co-designed alongside a specific enterprise hardware topology optimized for memory bandwidth and Rapid Packed Math (FP16):

* **Motherboard:** Supermicro Dual-EPYC (e.g., H11DSi) for a memory dense motherboard solution
* **System RAM:** 320GB+ (acting as the Holographic Dictionary pool) GGUF filesize x10 in RAM 
* **GPUs:** 3x AMD Radeon Pro V340 (6x Vega 10 / `gfx900` dies) with additional NVidia/Intel OpenCL support
* **Interconnect:** PCIe 3.0 x4 via shielded risers

---

## 🐳 Software Stack: Containerized Rusticl

HoloQubed does not use AMD's proprietary drivers. Instead, it runs inside a lightweight **Ubuntu 26.04 Docker container** leveraging **Mesa's Rusticl**, a modern, Rust-based OpenCL 3.0 implementation that fully supports `radeonsi` and Rapid Packed Math.

### Building the Environment
You can build the Docker container locally using the provided `Dockerfile`.

```bash
docker build -t twobombs/thereminq-holoqubed:latest .
```

## A note on NUMA nodes

If/as you are running HoloQubed on a dual-socket system (like AMD EPYC), you are dealing with Non-Uniform Memory Access (NUMA).

Because HoloQubed streams Zero-Copy coordinates directly from System RAM to the GPU via Direct Memory Access (DMA), crossing the Infinity Fabric between CPU nodes will cripple performance. Furthermore, dynamic OS-level memory balancing will cause PCIe bus timeouts.

How to run HoloQubed safely:

1. Disable OS Automatic Balancing:
You must turn off the host OS's dynamic NUMA balancer before running the container.

```bash
echo 0 | sudo tee /proc/sys/kernel/numa_balancing
sudo systemctl stop numad
```

2. Launch with Docker NUMA Pinning (--cpuset):
Use Docker's native cgroup controls to lock the container's memory pool and CPU threads strictly to the physical socket that controls your target GPU.

```bash
# Example: Running on GPU 0 (renderD128) bound strictly to CPU Node 0.
# The --ipc=host flag is critical for mapping large Zero-Copy buffers.
docker run -it --rm \
  --device=/dev/kfd \
  --device=/dev/dri/renderD128 \
  --group-add render \
  --group-add video \
  --cpuset-cpus="0-31" \
  --cpuset-mems="0" \
  --ipc=host \
  -v $(pwd):/app \
  twobombs/thereminq-holoqubed /bin/bash
```
For MultiGPU leverage `docker-compose.yml`
