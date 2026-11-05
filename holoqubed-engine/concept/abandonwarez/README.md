# ThereminQ Holoqubed Concept - Abandonwarez

This directory contains abandoned or older experimental concepts for the ThereminQ Holoqubed AI inference engine. These represent past attempts to integrate HoloQubed's sparse execution patterns with standard C++ ecosystems before the transition to direct PyTorch/HuggingFace C++ extensions.

## Subdirectories

*   `llama-integration/`: Older attempts at integrating HoloQubed with `llama.cpp`. These scripts attempted to patch `ggml` dynamically to recognize a custom `HOLO_SPARSE` data type and force the model into evaluating via a custom OpenCL backend injected natively into the `llama.cpp` tree.
