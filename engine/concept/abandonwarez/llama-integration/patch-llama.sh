# convert holo file to gguf masquerading holo file and injects holographic rendering in llama.cpp cpu module for native OpenCL actication

git clone https://github.com/ggml-org/llama.cpp.git
python3 pack_llama_holo.py
python3 hook_ggml_core.py
cd llama.cpp && cmake --build build --config Release -j 48 && cd ..
./llama.cpp/build/bin/llama-cli -m nanochat_holo_fused.gguf -p "there is no spoon" -n 40
