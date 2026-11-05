#include <torch/extension.h>
#include <CL/cl.h>
#include <iostream>

cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel spmv_kernel;
bool is_initialized = false;

// The Grid-Stride Local Reduction SpMV Kernel (Unchanged per your design requirements)
const char* SPMV_KERNEL_CODE = R"(
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

inline void atomic_add_local_float(volatile __local float *source, const float operand) {
    union { unsigned int int_val; float float_val; } newVal;
    union { unsigned int int_val; float float_val; } prevVal;
    do {
        prevVal.float_val = *source;
        newVal.float_val = prevVal.float_val + operand;
    } while (atomic_cmpxchg((volatile __local unsigned int *)source, prevVal.int_val, newVal.int_val) != prevVal.int_val);
}

inline void atomic_add_global_float(volatile __global float *source, const float operand) {
    union { unsigned int int_val; float float_val; } newVal;
    union { unsigned int int_val; float float_val; } prevVal;
    do {
        prevVal.float_val = *source;
        newVal.float_val = prevVal.float_val + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.int_val, newVal.int_val) != prevVal.int_val);
}

__kernel void spmv_holo_weights_fast(
    __global const ulong* morton_coords,   
    __global const ushort* sparse_weights_fp16, 
    __global const float* input_vector,
    __global float* output_vector,
    const int num_elements,
    const int out_features,
    __local float* local_out
) {
    int lid = get_local_id(0);
    int local_size = get_local_size(0);
    int gid = get_global_id(0);
    int global_size = get_global_size(0);

    // 1. Collaborative Zeroing of the Workgroup's L1 Output Cache
    for (int i = lid; i < out_features; i += local_size) {
        local_out[i] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2. Grid-Stride Loop: Threads chew through sparse chunks and add to L1 Cache
    for (int i = gid; i < num_elements; i += global_size) {
        ulong m_coord = morton_coords[i];
        float weight = vload_half(i, sparse_weights_fp16); 

        uint row = 0; uint col = 0;
        for (int bit = 0; bit < 16; bit++) {
            ulong shift = bit * 2; 
            row |= (uint)(((m_coord >> (0 + shift)) & 1) << bit);
            col |= (uint)(((m_coord >> (1 + shift)) & 1) << bit);
        }

        float activation = weight * input_vector[col];
        if (activation != 0.0f) {
            atomic_add_local_float(&local_out[row], activation);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 3. Collaborative Flush from L1 Cache back to Global VRAM
    for (int i = lid; i < out_features; i += local_size) {
        float val = local_out[i];
        if (val != 0.0f) {
            atomic_add_global_float(&output_vector[i], val);
        }
    }
}
)";

void init_opencl() {
    if (is_initialized) return;
    cl_uint numPlatforms; clGetPlatformIDs(1, NULL, &numPlatforms);
    cl_platform_id platform; clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device; clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device, 0, NULL);
    program = clCreateProgramWithSource(context, 1, &SPMV_KERNEL_CODE, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    
    spmv_kernel = clCreateKernel(program, "spmv_holo_weights_fast", NULL);
    is_initialized = true;
    std::cout << "[Holoqubed C++] Pure Sparse Engine with L1 Wavefront Reduction Initialized.\n";
}

class NativeHoloLayer {
    cl_mem coords_buf;
    cl_mem weights_buf;
    int in_features;
    int out_features;
    int num_elements;

public:
    NativeHoloLayer(torch::Tensor coords, torch::Tensor weights, int in_feat, int out_feat, int num_elem) {
        in_features = in_feat;
        out_features = out_feat;
        num_elements = num_elem;

        // VRAM Pinning (Pure Sparse) - Persistently stored for all threads to share
        coords_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_elements * sizeof(uint64_t), coords.data_ptr(), NULL);
        weights_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_elements * sizeof(uint16_t), weights.data_ptr(), NULL);
    }

    ~NativeHoloLayer() {
        clReleaseMemObject(coords_buf);
        clReleaseMemObject(weights_buf);
    }

    torch::Tensor forward(torch::Tensor input_vec) {
        auto input_contig = input_vec.contiguous();
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input_vec.device());
        auto output_vec = torch::zeros({out_features}, options);

        // FIX: Dynamically allocate I/O buffers per forward pass for native thread safety
        cl_mem in_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, in_features * sizeof(float), input_contig.data_ptr(), NULL);
        cl_mem out_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, out_features * sizeof(float), NULL, NULL);

        float zero = 0.0f;
        clEnqueueFillBuffer(queue, out_buf, &zero, sizeof(float), 0, out_features * sizeof(float), 0, NULL, NULL);

        // Architecting the Grid: 60 Compute Units * 4 Workgroups per CU = 240
        size_t local_size = 256;
        size_t global_size = 240 * local_size; 
        
        // Safety cap if layer is extremely small
        if (num_elements < global_size) {
            global_size = ((num_elements / local_size) + 1) * local_size;
        }

        clSetKernelArg(spmv_kernel, 0, sizeof(cl_mem), &coords_buf);
        clSetKernelArg(spmv_kernel, 1, sizeof(cl_mem), &weights_buf);
        clSetKernelArg(spmv_kernel, 2, sizeof(cl_mem), &in_buf);
        clSetKernelArg(spmv_kernel, 3, sizeof(cl_mem), &out_buf);
        clSetKernelArg(spmv_kernel, 4, sizeof(int), &num_elements);
        clSetKernelArg(spmv_kernel, 5, sizeof(int), &out_features);
        
        clSetKernelArg(spmv_kernel, 6, out_features * sizeof(float), NULL);

        clEnqueueNDRangeKernel(queue, spmv_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
        clEnqueueReadBuffer(queue, out_buf, CL_TRUE, 0, out_features * sizeof(float), output_vec.data_ptr(), 0, NULL, NULL);

        // FIX: Free dynamic thread-local buffers to prevent VRAM memory leak
        clReleaseMemObject(in_buf);
        clReleaseMemObject(out_buf);

        return output_vec;
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init_opencl", &init_opencl, "Initialize Native OpenCL Context");
    pybind11::class_<NativeHoloLayer>(m, "NativeHoloLayer")
        .def(pybind11::init<torch::Tensor, torch::Tensor, int, int, int>())
        .def("forward", &NativeHoloLayer::forward);
}
