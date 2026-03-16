# ThereminQ Holoqubed Concept

This directory contains conceptual prototypes for ThereminQ Holoqubed.

## Files

*   `gguf_vs_holo_divergences.py`: Accuracy Harness. Runs a dense GGUF reference model side-by-side with the sparse `.holo` engine to measure mathematical divergence caused by the SiLU sparsity threshold.
*   `holo_generate_hf.py`: Text Generation Engine. Monkey-patches a Hugging Face `PreTrainedModel`, replacing dense `nn.Linear` layers with custom `HoloLinear` PyTorch modules backed by the OpenCL SpMV engine.
*   `holoqubed_prototype.py`: PyOpenCL Prototype. A sparse, holographic AI inference engine bypassing dense matrix math via O(1) spatial queries and FP16 Rapid Packed Math. Target: AMD Radeon Pro V340 (Vega 10) via Mesa Rusticl.
