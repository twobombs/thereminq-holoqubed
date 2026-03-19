import os

path = "llama.cpp/ggml/src/ggml.c"
with open(path, "r") as f:
    data = f.read()

# Force the memory allocator to give us 8 bytes per element
data = data.replace(".type_size                = 1,", ".type_size                = 8,")
data = data.replace(".type_size                = 2,", ".type_size                = 8,")

with open(path, "w") as f:
    f.write(data)
print(" -> Updated HOLO_SPARSE type_size to 8!")
