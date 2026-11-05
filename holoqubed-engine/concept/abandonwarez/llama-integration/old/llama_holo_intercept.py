import os

print("Injecting Holoqubed OpenCL Engine into llama.cpp...")

REPO_DIR = "llama.cpp"

# First, find exactly where ggml-cpu.c lives in this specific fork
cpu_c_path = None
for root, dirs, files in os.walk(REPO_DIR):
    if "ggml-cpu.c" in files:
        cpu_c_path = os.path.join(root, "ggml-cpu.c")
        break

if not cpu_c_path:
    print(" [ERROR] Could not find ggml-cpu.c")
    exit(1)

# Write the OpenCL backend right next to ggml-cpu.c
holo_c_path = os.path.join(os.path.dirname(cpu_c_path), "ggml-holo.c")

holo_c_content = """
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "ggml.h"

static cl_context holo_ctx;
static cl_command_queue holo_queue;
static cl_program holo_prog;
static cl_kernel holo_kernel;
static bool holo_init = false;

const char* SPMV_KERNEL_CODE = 
"#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable\\n"
"#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\\n"
"inline void atomic_add_local_float(volatile __local float *source, const float operand) {\\n"
"    union { unsigned int int_val; float float_val; } newVal;\\n"
"    union { unsigned int int_val; float float_val; } prevVal;\\n"
"    do {\\n"
"        prevVal.float_val = *source;\\n"
"        newVal.float_val = prevVal.float_val + operand;\\n"
"    } while (atomic_cmpxchg((volatile __local unsigned int *)source, prevVal.int_val, newVal.int_val) != prevVal.int_val);\\n"
"}\\n"
"inline void atomic_add_global_float(volatile __global float *source, const float operand) {\\n"
"    union { unsigned int int_val; float float_val; } newVal;\\n"
"    union { unsigned int int_val; float float_val; } prevVal;\\n"
"    do {\\n"
"        prevVal.float_val = *source;\\n"
"        newVal.float_val = prevVal.float_val + operand;\\n"
"    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.int_val, newVal.int_val) != prevVal.int_val);\\n"
"}\\n"
"__kernel void spmv_holo_weights_fast(\\n"
"    __global const ulong* morton_coords,\\n"
"    __global const ushort* sparse_weights_fp16,\\n"
"    __global const float* input_vector,\\n"
"    __global float* output_vector,\\n"
"    const int num_elements,\\n"
"    const int out_features,\\n"
"    __local float* local_out\\n"
") {\\n"
"    int lid = get_local_id(0);\\n"
"    int local_size = get_local_size(0);\\n"
"    int gid = get_global_id(0);\\n"
"    int global_size = get_global_size(0);\\n"
"    for (int i = lid; i < out_features; i += local_size) {\\n"
"        local_out[i] = 0.0f;\\n"
"    }\\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\\n"
"    for (int i = gid; i < num_elements; i += global_size) {\\n"
"        ulong m_coord = morton_coords[i];\\n"
"        float weight = vload_half(i, sparse_weights_fp16);\\n"
"        uint row = 0; uint col = 0;\\n"
"        for (int bit = 0; bit < 16; bit++) {\\n"
"            ulong shift = bit * 2;\\n"
"            row |= (uint)(((m_coord >> (0 + shift)) & 1) << bit);\\n"
"            col |= (uint)(((m_coord >> (1 + shift)) & 1) << bit);\\n"
"        }\\n"
"        float activation = weight * input_vector[col];\\n"
"        if (activation != 0.0f) {\\n"
"            atomic_add_local_float(&local_out[row], activation);\\n"
"        }\\n"
"    }\\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\\n"
"    for (int i = lid; i < out_features; i += local_size) {\\n"
"        float val = local_out[i];\\n"
"        if (val != 0.0f) {\\n"
"            atomic_add_global_float(&output_vector[i], val);\\n"
"        }\\n"
"    }\\n"
"}\\n";

struct holo_extra {
    cl_mem coords_buf;
    cl_mem weights_buf;
};

void ggml_holo_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {
    if (!holo_init) {
        cl_uint numPlatforms; clGetPlatformIDs(1, NULL, &numPlatforms);
        cl_platform_id platform; clGetPlatformIDs(1, &platform, NULL);
        cl_device_id device; clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        holo_ctx = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
        holo_queue = clCreateCommandQueue(holo_ctx, device, 0, NULL);
        holo_prog = clCreateProgramWithSource(holo_ctx, 1, &SPMV_KERNEL_CODE, NULL, NULL);
        clBuildProgram(holo_prog, 1, &device, NULL, NULL, NULL);
        holo_kernel = clCreateKernel(holo_prog, "spmv_holo_weights_fast", NULL);
        holo_init = true;
        printf("\\n[Holoqubed] Pure Sparse OpenCL Intercept Activated inside llama.cpp!\\n");
    }

    uint64_t num_elements = *(uint64_t*)src0->data;
    const uint64_t* coords = (const uint64_t*)((char*)src0->data + 8);
    const uint16_t* weights = (const uint16_t*)((char*)src0->data + 8 + num_elements * 8);

    // ZERO-COPY PINNING: We cache the 3.5GB of VRAM directly on the llama.cpp tensor struct!
    if (src0->extra == NULL) {
        struct holo_extra * ex = (struct holo_extra *)malloc(sizeof(struct holo_extra));
        ex->coords_buf = clCreateBuffer(holo_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_elements * sizeof(uint64_t), (void*)coords, NULL);
        ex->weights_buf = clCreateBuffer(holo_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_elements * sizeof(uint16_t), (void*)weights, NULL);
        ((struct ggml_tensor *)src0)->extra = ex; // Bypass const
    }

    struct holo_extra * ex = (struct holo_extra *)src0->extra;

    int in_features = src1->ne[0];
    int out_features = dst->ne[0];
    int num_rows = src1->ne[1]; 
    
    for (int r = 0; r < num_rows; r++) {
        float* in_ptr = (float*)src1->data + (r * in_features);
        float* out_ptr = (float*)dst->data + (r * out_features);

        cl_mem in_buf = clCreateBuffer(holo_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, in_features * sizeof(float), in_ptr, NULL);
        cl_mem out_buf = clCreateBuffer(holo_ctx, CL_MEM_READ_WRITE, out_features * sizeof(float), NULL, NULL);

        float zero = 0.0f;
        clEnqueueFillBuffer(holo_queue, out_buf, &zero, sizeof(float), 0, out_features * sizeof(float), 0, NULL, NULL);

        size_t local_size = 256;
        size_t global_size = 240 * local_size;
        if (num_elements < global_size) {
            global_size = ((num_elements / local_size) + 1) * local_size;
        }

        int ne_int = (int)num_elements;
        clSetKernelArg(holo_kernel, 0, sizeof(cl_mem), &ex->coords_buf);
        clSetKernelArg(holo_kernel, 1, sizeof(cl_mem), &ex->weights_buf);
        clSetKernelArg(holo_kernel, 2, sizeof(cl_mem), &in_buf);
        clSetKernelArg(holo_kernel, 3, sizeof(cl_mem), &out_buf);
        clSetKernelArg(holo_kernel, 4, sizeof(int), &ne_int);
        clSetKernelArg(holo_kernel, 5, sizeof(int), &out_features);
        clSetKernelArg(holo_kernel, 6, out_features * sizeof(float), NULL);

        clEnqueueNDRangeKernel(holo_queue, holo_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
        clEnqueueReadBuffer(holo_queue, out_buf, CL_TRUE, 0, out_features * sizeof(float), out_ptr, 0, NULL, NULL);

        clReleaseMemObject(in_buf);
        clReleaseMemObject(out_buf);
    }
}
"""

with open(holo_c_path, "w") as f:
    f.write(holo_c_content)
print(f" -> Created C OpenCL Backend at {holo_c_path}")

with open(cpu_c_path, "r") as f:
    cpu_data = f.read()

# Safely inject AFTER ggml.h so the structs are fully defined
if '#include "ggml-holo.c"' not in cpu_data:
    cpu_data = cpu_data.replace('#include "ggml.h"', '#include "ggml.h"\n#include "ggml-holo.c"')

hook_target = "if (src0->type == GGML_TYPE_F16"
hook_replacement = """if (src0->type == GGML_TYPE_HOLO_SPARSE) {
        ggml_holo_mul_mat(src0, src1, dst);
        return;
    }

    if (src0->type == GGML_TYPE_F16"""

if "ggml_holo_mul_mat" not in cpu_data:
    cpu_data = cpu_data.replace(hook_target, hook_replacement)
    with open(cpu_c_path, "w") as f:
        f.write(cpu_data)
    print(f" -> Successfully wired OpenCL tripwire into {cpu_c_path}!")
else:
    print(f" -> {cpu_c_path} is already intercepted.")
