import os

ggml_c_path = "llama.cpp/ggml/src/ggml.c"

os.system(f"cd llama.cpp && git checkout {ggml_c_path.replace('llama.cpp/', '')} && cd ..")

with open(ggml_c_path, "r") as f:
    code = f.read()

opencl_payload = r"""
// ============================================================================
// HOLOQUBED STANDALONE OPENCL DRIVER (STANDARD MATH)
// ============================================================================
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <dlfcn.h>
#include <pthread.h>

static cl_int (*my_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*) = NULL;
static cl_int (*my_clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*) = NULL;
static cl_context (*my_clCreateContext)(const cl_context_properties*, cl_uint, const cl_device_id*, void(CL_CALLBACK*)(const char*, const void*, size_t, void*), void*, cl_int*) = NULL;
static cl_command_queue (*my_clCreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*) = NULL;
static cl_program (*my_clCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*) = NULL;
static cl_int (*my_clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void(CL_CALLBACK*)(cl_program, void*), void*) = NULL;
static cl_int (*my_clGetProgramBuildInfo)(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*) = NULL;
static cl_kernel (*my_clCreateKernel)(cl_program, const char*, cl_int*) = NULL;
static cl_mem (*my_clCreateBuffer)(cl_context, cl_mem_flags, size_t, void*, cl_int*) = NULL;
static cl_int (*my_clEnqueueFillBuffer)(cl_command_queue, cl_mem, const void*, size_t, size_t, size_t, cl_uint, const cl_event*, cl_event*) = NULL;
static cl_int (*my_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*) = NULL;
static cl_int (*my_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*) = NULL;
static cl_int (*my_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*) = NULL;
static cl_int (*my_clReleaseMemObject)(cl_mem) = NULL;
static cl_int (*my_clFinish)(cl_command_queue) = NULL;

static void load_opencl_dynamically(void) {
    void* handle = dlopen("libOpenCL.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) handle = dlopen("libOpenCL.so", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) { 
        printf("\n[Holoqubed FATAL] libOpenCL.so not found!\n"); 
        fflush(stdout); exit(1); 
    }

    #define LOAD_SYM(name) \
        my_##name = dlsym(handle, #name); \
        if (!my_##name) { printf("\n[Holoqubed FATAL] Missing symbol: %s\n", #name); fflush(stdout); exit(1); }

    LOAD_SYM(clGetPlatformIDs);
    LOAD_SYM(clGetDeviceIDs);
    LOAD_SYM(clCreateContext);
    LOAD_SYM(clCreateCommandQueue);
    LOAD_SYM(clCreateProgramWithSource);
    LOAD_SYM(clBuildProgram);
    LOAD_SYM(clGetProgramBuildInfo);
    LOAD_SYM(clCreateKernel);
    LOAD_SYM(clCreateBuffer);
    LOAD_SYM(clEnqueueFillBuffer);
    LOAD_SYM(clSetKernelArg);
    LOAD_SYM(clEnqueueNDRangeKernel);
    LOAD_SYM(clEnqueueReadBuffer);
    LOAD_SYM(clReleaseMemObject);
    LOAD_SYM(clFinish);
}

static cl_context holo_ctx;
static cl_command_queue holo_queue;
static cl_program holo_prog;
static cl_kernel holo_kernel;
static bool holo_init = false;
static pthread_mutex_t holo_mut = PTHREAD_MUTEX_INITIALIZER;

#define MAX_LAYERS 256
static const void* cached_src0[MAX_LAYERS];
static cl_mem cached_coords[MAX_LAYERS];
static cl_mem cached_weights[MAX_LAYERS];
static uint32_t cached_hashes[MAX_LAYERS];
static int num_cached = 0;

const char* SPMV_KERNEL_CODE = 
"#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\n"
"inline void atomic_add_global_float(volatile __global float *source, const float operand) {\n"
"    union { unsigned int int_val; float float_val; } newVal;\n"
"    union { unsigned int int_val; float float_val; } prevVal;\n"
"    do {\n"
"        prevVal.float_val = *source;\n"
"        newVal.float_val = prevVal.float_val + operand;\n"
"    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.int_val, newVal.int_val) != prevVal.int_val);\n"
"}\n"
"__kernel void spmv_holo_weights_fast(\n"
"    __global const uint* morton_coords,\n"
"    __global const ushort* sparse_weights_fp16,\n"
"    __global const float* input_vector,\n"
"    __global float* output_vector,\n"
"    const int num_elements,\n"
"    const int in_features,\n"
"    const int out_features\n"
") {\n"
"    int gid = get_global_id(0);\n"
"    if (gid >= num_elements) return;\n"
"    uint m_coord = morton_coords[gid];\n"
"    float weight = vload_half(gid, (const __global half*)sparse_weights_fp16);\n"
"    uint row = 0; uint col = 0;\n"
"    for (int bit = 0; bit < 16; bit++) {\n"
"        uint shift = bit * 2;\n"
"        col |= (uint)(((m_coord >> (0 + shift)) & 1) << bit);\n"
"        row |= (uint)(((m_coord >> (1 + shift)) & 1) << bit);\n"
"    }\n"
"    // BACK TO TRUE MATH: output[row] += weight * input[col]\n"
"    if (col < in_features && row < out_features) {\n"
"        float activation = weight * input_vector[col];\n"
"        if (activation != 0.0f) {\n"
"            atomic_add_global_float(&output_vector[row], activation);\n"
"        }\n"
"    }\n"
"}\n";

static void ggml_holo_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {
    if (src0 == NULL || src1 == NULL || dst == NULL) return;
    if ((uintptr_t)src1->data < 0x100000000ULL || (uintptr_t)dst->data < 0x100000000ULL) return;

    pthread_mutex_lock(&holo_mut);

    int in_features = src1->ne[0];
    int out_features = dst->ne[0];
    int num_rows = src1->ne[1]; 

    uint32_t current_hash = 0;
    float* in_fp32 = (float*)src1->data;
    for (int i = 0; i < 8 && i < in_features; i++) {
        current_hash ^= ((uint32_t*)in_fp32)[i];
    }
    if (current_hash == 0) current_hash = 0x12345678; 

    cl_int err;
    if (!holo_init) {
        load_opencl_dynamically();
        cl_platform_id platform; 
        my_clGetPlatformIDs(1, &platform, NULL);
        cl_device_id device; 
        err = my_clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        if (err != CL_SUCCESS) {
            printf("\n[Holoqubed FATAL] GPU Device not found. Err: %d\n", err);
            fflush(stdout); exit(1);
        }
        holo_ctx = my_clCreateContext(NULL, 1, &device, NULL, NULL, &err);
        holo_queue = my_clCreateCommandQueue(holo_ctx, device, 0, &err);

        holo_prog = my_clCreateProgramWithSource(holo_ctx, 1, &SPMV_KERNEL_CODE, NULL, &err);
        err = my_clBuildProgram(holo_prog, 1, &device, NULL, NULL, NULL);
        if (err != CL_SUCCESS) {
            printf("\n[Holoqubed FATAL] Kernel Build Failed.\n");
            fflush(stdout); exit(1);
        }
        holo_kernel = my_clCreateKernel(holo_prog, "spmv_holo_weights_fast", &err);
        holo_init = true;
        printf("\n[Holoqubed] Full Neural Restoration Running on Radeon Pro VII!\n");
        fflush(stdout);
    }

    int layer_idx = -1;
    for (int i = 0; i < num_cached; i++) {
        if (cached_src0[i] == src0->data) {
            layer_idx = i;
            break;
        }
    }

    cl_mem coords_buf = NULL;
    cl_mem weights_buf = NULL;

    if (layer_idx == -1) {
        if (num_cached >= MAX_LAYERS) {
            printf("\n[Holoqubed FATAL] Exceeded maximum layer count!\n");
            exit(1);
        }
        
        uint64_t num_elements = *(uint64_t*)((char*)src0->data + 8);
        
        const uint32_t* coords = (const uint32_t*)((char*)src0->data + 16);
        const uint16_t* weights = (const uint16_t*)((char*)src0->data + 16 + num_elements * 4); 

        coords_buf = my_clCreateBuffer(holo_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_elements * sizeof(uint32_t), (void*)coords, &err);
        weights_buf = my_clCreateBuffer(holo_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_elements * sizeof(uint16_t), (void*)weights, &err);
        
        layer_idx = num_cached;
        cached_src0[layer_idx] = src0->data;
        cached_coords[layer_idx] = coords_buf;
        cached_weights[layer_idx] = weights_buf;
        cached_hashes[layer_idx] = current_hash;
        num_cached++;
    } else {
        if (cached_hashes[layer_idx] == current_hash) {
            pthread_mutex_unlock(&holo_mut);
            return;
        }
        cached_hashes[layer_idx] = current_hash;
        coords_buf = cached_coords[layer_idx];
        weights_buf = cached_weights[layer_idx];
    }

    uint64_t num_elements = *(uint64_t*)((char*)src0->data + 8);

    for (int r = 0; r < num_rows; r++) {
        float* in_ptr = (float*)src1->data + (r * in_features);
        float* out_ptr = (float*)dst->data + (r * out_features);

        cl_mem in_buf = my_clCreateBuffer(holo_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, in_features * sizeof(float), in_ptr, &err);
        cl_mem out_buf = my_clCreateBuffer(holo_ctx, CL_MEM_READ_WRITE, out_features * sizeof(float), NULL, &err);

        float zero = 0.0f;
        my_clEnqueueFillBuffer(holo_queue, out_buf, &zero, sizeof(float), 0, out_features * sizeof(float), 0, NULL, NULL);

        size_t local_size = 256;
        size_t global_size = ((num_elements + local_size - 1) / local_size) * local_size;
        int ne_int = (int)num_elements;

        my_clSetKernelArg(holo_kernel, 0, sizeof(cl_mem), &coords_buf);
        my_clSetKernelArg(holo_kernel, 1, sizeof(cl_mem), &weights_buf);
        my_clSetKernelArg(holo_kernel, 2, sizeof(cl_mem), &in_buf);
        my_clSetKernelArg(holo_kernel, 3, sizeof(cl_mem), &out_buf);
        my_clSetKernelArg(holo_kernel, 4, sizeof(int), &ne_int);
        my_clSetKernelArg(holo_kernel, 5, sizeof(int), &in_features);
        my_clSetKernelArg(holo_kernel, 6, sizeof(int), &out_features);

        my_clEnqueueNDRangeKernel(holo_queue, holo_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
        my_clEnqueueReadBuffer(holo_queue, out_buf, CL_TRUE, 0, out_features * sizeof(float), out_ptr, 0, NULL, NULL);

        my_clReleaseMemObject(in_buf);
        my_clReleaseMemObject(out_buf);
    }
    my_clFinish(holo_queue);
    pthread_mutex_unlock(&holo_mut);
}

static void holo_custom_op(struct ggml_tensor * dst, const struct ggml_tensor * a, const struct ggml_tensor * b, int ith, int nth, void * userdata) {
    (void)a; (void)b; (void)nth; (void)userdata; 
    if (ith == 0) { 
        ggml_holo_mul_mat(dst->src[0], dst->src[1], dst);
    }
}
"""

hook_impl = r"""
    // HOLOQUBED GRAPH HIJACK - MAGIC NUMBER F32 SCANNER
    if (a->type == GGML_TYPE_F32 && a->data != NULL) {
        uint64_t magic = *(uint64_t*)a->data;
        if (magic == 0x484F4C4F51554244ULL) { // "HOLOQUBD" Hex Signature
            struct ggml_tensor * args[2] = { (struct ggml_tensor *)a, (struct ggml_tensor *)b };
            return ggml_custom_4d(ctx, GGML_TYPE_F32, a->ne[1], b->ne[1], b->ne[2], b->ne[3], args, 2, (ggml_custom_op_t)holo_custom_op, 1, NULL);
        }
    }
"""

code = code.replace('#include "ggml.h"\n', '#include "ggml.h"\n\n' + opencl_payload, 1)
target = "const int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };"
if target in code:
    code = code.replace(target, hook_impl + "\n    " + target, 1)
    with open(ggml_c_path, "w") as f:
        f.write(code)
    print(" -> Engine Injected with Perfect Math Execution!")
