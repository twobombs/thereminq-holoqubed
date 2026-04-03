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

*   `gguf2holo.py`: Offline Model Ingress & Conversion Pipeline. Converts dense `.gguf` and `.pt` models (FP32/FP16/BF16) into the sparse, spatially encoded `.holo` dictionary. It implements a Hybrid Dynamic Scalpel using both distribution-blind (standard deviation) and distribution-aware (percentile) thresholds to prune weights safely. It utilizes batched Morton (Z-order) bit-interleaving via `np.bitwise_xor.reduce` for dynamic 1D spatial encoding. To combat memory usage, it processes tensors in parallel streams while protecting against OOM, intercepting early tensors to train a global Zstandard (`zstd`) dictionary which is injected directly into the final ZIP archive for highly optimized compression ratios.
*   `holo_loader.py`: Ingress Loader & JIT CPU Query Planner. Exposes the `HoloQueryPlanner` which mounts the compressed `.holo` dictionary directly from disk using `zipfile`. It extracts the embedded Zstandard dictionary to rapidly decompress payload layers on the fly. It performs ultra-fast O(log N) pathway lookups via `np.searchsorted` to extract only the active spatial coordinates and weights required for the given execution frame.
*   `holo_ext.cpp`: Native OpenCL C++ Extension. Provides pre-inflated native OpenCL execution kernels for scattering sparse weights and performing SpMV dense math. It pins the decompressed `coords` and `weights` to persistent VRAM buffers on OpenCL initialization. During the forward pass, it manages dynamic OpenCL local memory (`__local` cache) based on the output feature size and executes a highly optimized Grid-Stride Local Reduction SpMV kernel (`spmv_holo_weights_fast`) using OpenCL atomics. It integrates seamlessly into PyTorch via `pybind11`.
*   `holo_generate_ext.py`: Pure OpenCL Text Generation Edition. Monkey-patches a Hugging Face `PreTrainedModel` by replacing dense linear layers with a custom `HoloLinear` module. This module leverages `holo_ext` to scatter sparse Morton coordinates into persistent OpenCL VRAM buffers during initialization. Execution during text generation happens purely via the OpenCL PyTorch C++ extension, radically bypassing CPU execution overhead for Hugging Face models.
*   `holo_check.py`: Ingress Loader & Simulator. Simulates the CPU Query Planner loading layers from a `.holo` dictionary and executes a direct OpenCL Sparse Matrix-Vector Multiplication (SpMV) in VRAM using a built-in OpenCL compilation chain via `pyopencl`. It validates OpenCL functionality and simulates memory utilization savings.
*   `holo_setup.py`: Standard setup script for building and installing the `holo_ext` C++ PyTorch extension using `torch.utils.cpp_extension`.
*   `holo_setup.sh`: Shell script wrapper to automate the execution of `holo_setup.py` and environment preparation.
