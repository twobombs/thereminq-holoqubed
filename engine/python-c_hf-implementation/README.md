<img width="1408" height="768" alt="1774127890823" src="https://github.com/user-attachments/assets/ec8245b7-b7cf-42cf-9913-fc5a16beae34" />

**works with gguf and d34 models**

gguf/d34 holo graphic conversion from pt or gguf file
- default is set to a fairly agressive 0.5 to test the cut 
- you might want to put this at a lower level like 0.1 for more coherency

<img width="512" height="310" alt="Screenshot from 2026-03-16 10-17-25" src="https://github.com/user-attachments/assets/f9a81dba-d9bb-4efe-a950-326cb570189f" />

---

holo loader module 
- also used by the inference script to load the model

<img width="1074" height="683" alt="Screenshot from 2026-03-16 13-21-04" src="https://github.com/user-attachments/assets/4db458ce-56f6-49a8-9f04-5a00c3ae7ae6" />

---

holo generate ext 
- the actual sparse inference script that loads and executes the query with some temperature and top variables that are optional

<img width="1910" height="635" alt="Screenshot from 2026-03-21 09-35-41" src="https://github.com/user-attachments/assets/d581bb07-71b1-469f-aedf-055a7f238b13" />

python overhead visible in benchmark results 

<img width="1915" height="75" alt="Screenshot from 2026-03-21 09-46-40" src="https://github.com/user-attachments/assets/cfa6411f-681d-49cf-b0b4-76d93e3c56ee" />


## Files

*   `gguf2holo.py`: Offline Model Ingress & Conversion Pipeline. Converts dense `.gguf` and `.pt` models (FP32/FP16/BF16) into the sparse, spatially encoded `.holo` dictionary. Features parallel streaming, BF16 bit-shifting, OOM protection, & Zstd compression.
*   `holo_ext.cpp`: Native OpenCL C++ Extension. Provides pre-inflated native OpenCL execution kernels for scattering sparse weights and performing GEMV dense math.
*   `holo_generate_ext.py`: Pure OpenCL Text Generation Edition. Uses `holo_ext` to scatter sparse Morton coordinates into dense OpenCL VRAM buffers during initialization, and executes purely via OpenCL GEMV kernels to bypass CPU limitations for Hugging Face models.
*   `holo_loader.py`: Ingress Loader & CPU Query Planner. Streams Zstd-compressed layers from the `.holo` ZIP archive, performs ultra-fast O(log N) pathway lookups, and provides the CPU interface for fetching spatial coordinates.
*   `holo_setup.py`: Standard setup script for building and installing the `holo_ext` C++ PyTorch extension.
*   `holo_setup.sh`: Shell script wrapper to automate the execution of `holo_setup.py` and environment preparation.
