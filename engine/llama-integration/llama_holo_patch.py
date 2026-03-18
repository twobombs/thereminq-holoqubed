import os

print("Starting Holoqubed GGML Patcher...")

REPO_DIR = "llama.cpp"
ggml_h_path = os.path.join(REPO_DIR, "ggml/include/ggml.h")
ggml_c_path = os.path.join(REPO_DIR, "ggml/src/ggml.c")

# 1. Patch ggml.h
if not os.path.exists(ggml_h_path):
    print(f" [ERROR] Could not find {ggml_h_path}.")
    exit(1)

with open(ggml_h_path, "r") as f:
    h_data = f.read()

h_target = "GGML_TYPE_COUNT   = 41,"
h_replacement = "GGML_TYPE_HOLO_SPARSE = 41,\n        GGML_TYPE_COUNT   = 42,"

if "GGML_TYPE_HOLO_SPARSE" not in h_data:
    if h_target in h_data:
        with open(ggml_h_path, "w") as f:
            f.write(h_data.replace(h_target, h_replacement))
        print(f" -> Successfully patched {ggml_h_path}")
    else:
        print(f" -> [WARNING] Could not find {h_target} in {ggml_h_path}.")
else:
    print(f" -> {ggml_h_path} is already patched")

# 2. Patch ggml.c
if not os.path.exists(ggml_c_path):
    print(f" [ERROR] Could not find {ggml_c_path}.")
    exit(1)

with open(ggml_c_path, "r") as f:
    c_data = f.read()

c_target = """    [38] = { // GGML_TYPE_IQ4_NL_8_8
        .type_name                = "TYPE_IQ4_NL_8_8 REMOVED, use IQ4_NL with runtime repacking",
        .blck_size                = 0,
        .type_size                = 0,
        .is_quantized             = false,
    },
};"""

c_replacement = """    [38] = { // GGML_TYPE_IQ4_NL_8_8
        .type_name                = "TYPE_IQ4_NL_8_8 REMOVED, use IQ4_NL with runtime repacking",
        .blck_size                = 0,
        .type_size                = 0,
        .is_quantized             = false,
    },
    [GGML_TYPE_HOLO_SPARSE] = {
        .type_name                = "HOLO_SPARSE",
        .blck_size                = 1,
        .type_size                = 1,
        .is_quantized             = false,
    },
};"""

if "HOLO_SPARSE" not in c_data:
    if c_target in c_data:
        with open(ggml_c_path, "w") as f:
            f.write(c_data.replace(c_target, c_replacement))
        print(f" -> Successfully patched {ggml_c_path}")
    else:
        print(f" -> [WARNING] Could not find the type_traits array end in {ggml_c_path}.")
else:
    print(f" -> {ggml_c_path} is already patched")
