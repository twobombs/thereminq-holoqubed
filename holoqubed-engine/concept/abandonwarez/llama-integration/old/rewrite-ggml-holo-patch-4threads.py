import os

holo_c_path = "llama.cpp/ggml/src/ggml-cpu/ggml-holo.c"

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
    // 1. THE MUTEX: Prevent 48-thread collision during Kernel initialization and argument setting
    #pragma omp critical
    {
        if (!holo_init) {
            cl_uint numPlatforms = 0; 
            clGetPlatformIDs(1, NULL, &numPlatforms);
            if (numPlatforms == 0) {
                printf("\\n[Holoqubed FATAL] No OpenCL Platforms found! Install PoCL or an OpenCL driver.\\n");
                exit(1);
            }
            
            cl_platform_id platform; 
            clGetPlatformIDs(1, &platform, NULL);
            
            cl_device_id device; 
            // 2. DEVICE FALLBACK: Use ALL devices (CPU/GPU) instead of crashing on missing GPU
            cl_int err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
            if (err != CL_SUCCESS) {
                printf("\\n[Holoqubed FATAL] OpenCL Device lookup failed. Error code: %d\\n", err);
                exit(1);
            }
            
            holo_ctx = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
            holo_queue = clCreateCommandQueue(holo_ctx, device, 0, NULL);
            holo_prog = clCreateProgramWithSource(holo_ctx, 1, &SPMV_KERNEL_CODE, NULL, NULL);
            clBuildProgram(holo_prog, 1, &device, NULL, NULL, NULL);
            holo_kernel = clCreateKernel(holo_prog, "spmv_holo_weights_fast", NULL);
            holo_init = true;
            printf("\\n[Holoqubed] Thread-Safe OpenCL Intercept Activated!\\n");
        }

        uint64_t num_elements = *(uint64_t*)src0->data;
        const uint64_t* coords = (const uint64_t*)((char*)src0->data + 8);
        const uint16_t* weights = (const uint16_t*)((char*)src0->data + 8 + num_elements * 8);

        if (src0->extra == NULL) {
            struct holo_extra * ex = (struct holo_extra *)malloc(sizeof(struct holo_extra));
            ex->coords_buf = clCreateBuffer(holo_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_elements * sizeof(uint64_t), (void*)coords, NULL);
            ex->weights_buf = clCreateBuffer(holo_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_elements * sizeof(uint16_t), (void*)weights, NULL);
            ((struct ggml_tensor *)src0)->extra = ex;
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
    } // END PRAGMA OMP CRITICAL LOCK
}
"""

with open(holo_c_path, "w") as f:
    f.write(holo_c_content)
print(" -> Successfully locked OpenCL backend for multi-threaded safety!")
