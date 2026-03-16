works with gguf and d34 models

gguf/d34 holo graphic conversion

<img width="512" height="310" alt="Screenshot from 2026-03-16 10-17-25" src="https://github.com/user-attachments/assets/f9a81dba-d9bb-4efe-a950-326cb570189f" />

holo loader module

<img width="1074" height="683" alt="Screenshot from 2026-03-16 13-21-04" src="https://github.com/user-attachments/assets/4db458ce-56f6-49a8-9f04-5a00c3ae7ae6" />

holo generate ext 

<img width="694" height="383" alt="Screenshot from 2026-03-16 18-24-11" src="https://github.com/user-attachments/assets/003d67bf-9394-4396-b844-4a2c59fcac0e" />

python overhead visible in benchmark results 

## Files

*   `gguf2holo.py`: Offline Model Ingress & Conversion Pipeline. Converts dense `.gguf` and `.pt` models (FP32/FP16/BF16) into the sparse, spatially encoded `.holo` dictionary. Features parallel streaming, BF16 bit-shifting, OOM protection, & Zstd compression.
*   `holo_ext.cpp`: Native OpenCL C++ Extension. Provides pre-inflated native OpenCL execution kernels for scattering sparse weights and performing GEMV dense math.
*   `holo_generate_ext.py`: Pure OpenCL Text Generation Edition. Uses `holo_ext` to scatter sparse Morton coordinates into dense OpenCL VRAM buffers during initialization, and executes purely via OpenCL GEMV kernels to bypass CPU limitations for Hugging Face models.
*   `holo_loader.py`: Ingress Loader & CPU Query Planner. Streams Zstd-compressed layers from the `.holo` ZIP archive, performs ultra-fast O(log N) pathway lookups, and provides the CPU interface for fetching spatial coordinates.
*   `holo_setup.py`: Standard setup script for building and installing the `holo_ext` C++ PyTorch extension.
*   `holo_setup.sh`: Shell script wrapper to automate the execution of `holo_setup.py` and environment preparation.