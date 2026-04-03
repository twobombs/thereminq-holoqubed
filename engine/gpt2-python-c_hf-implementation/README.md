# GPT2 Python/C++ HF Implementation

This directory contains the Python and C++ implementation for integrating HoloQubed with Hugging Face GPT-2 models.

## Files
* `geometry_forge.py`: Transforms dense Flatland AI models into Complex Hilbert Phase Space for GPT-2. It maps the 2D Cartesian coordinates into a 1D continuous spatial distance using a JIT-compiled Hilbert Curve engine. It can inject a determininstic spatial phase shift, casting real-valued magnitudes into Complex Cartesian space (Real + Imaginary), preparing them for phase modulation during SpMV execution.
* `holo_ext.cpp`: Native OpenCL C++ Extension for GPT-2 models. It acts as a multi-GPU JIT Rehydrator. It loads the sparse quantum weights, reconstructs the original signed amplitudes, modulates them using spatial frequency interference, and outputs a dense matrix representation, effectively bypassing atomic operations entirely via static pre-allocation.
* `holo_generate_ext.py`: Text generation engine using OpenCL extension for GPT-2. Monkey-patches the Hugging Face `PreTrainedModel` replacing dense linear layers with custom OpenCL-backed `HoloLinear` implementations.
* `holo_loader.py`: Ingress loader and CPU Query Planner. Exposes the `HoloQueryPlanner` which mounts the compressed `.holo` dictionary directly from disk using `zipfile`.
* `holo_setup.py`: Standard setup script for building and installing the C++ PyTorch extension using `torch.utils.cpp_extension`.
