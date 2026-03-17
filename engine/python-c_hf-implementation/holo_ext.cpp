#include <torch/extension.h>
#include <CL/cl.h>
#include <iostream>

cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel spmv_kernel;
bool is_initialized = false;

// The True End-to-End Sparse SpMV Kernel
const char* SPMV_KERNEL_CODE = R"(
inline void atomic_add_float(volatile __global float *source, const float operand) {
    union { unsigned int int_val; float float_val; } newVal;
    union { unsigned int int_val; float float_val; } prevVal;
    do {
        prevVal.float_val = *source;
        newVal.float_val = prevVal.float_val + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.int_val, newVal.int_val) != prevVal.int_val);
}

__kernel void spmv_holo_weights(__global const ulong* morton_coords, __global const ushort* sparse_weights, __global const float* input_vector, __global float* output_vector, const int num_elements) {
    int gid = get_global_id(0);
    if (gid >= num_elements) return;

    ulong m_coord = morton_coords[gid];
    float weight = vload_half(gid, sparse_weights); 

    uint row = 0; uint col = 0;
    for (int bit = 0; bit < 16; bit++) {
        ulong shift = bit * 2; 
        row |= (uint)(((m_coord >> (0 + shift)) & 1) << bit);
        col |= (uint)(((m_coord >> (1 + shift)) & 1) << bit);
    }

    float activation = weight * input_vector[col];
    
    // Pure sparse execution: skip the write if activation is zero
    if (activation != 0.0f) {
        atomic_add_float(&output_vector[row], activation);
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
    
    spmv_kernel = clCreateKernel(program, "spmv_holo_weights", NULL);
    is_initialized = true;
    std::cout << "[Holoqubed C++] Pure Sparse-Serving OpenCL Engine Initialized.\n";
}

class NativeHoloLayer {
    cl_mem coords_buf;
    cl_mem weights_buf;
    cl_mem in_buf;
    cl_mem out_buf;
    int in_features;
    int out_features;
    int num_elements;

public:
    NativeHoloLayer(torch::Tensor coords, torch::Tensor weights, int in_feat, int out_feat, int num_elem) {
        in_features = in_feat;
        out_features = out_feat;
        num_elements = num_elem;

        // Permanently pin the SPARSE data in VRAM. No dense inflation.
        coords_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_elements * sizeof(uint64_t), coords.data_ptr(), NULL);
        weights_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_elements * sizeof(uint16_t), weights.data_ptr(), NULL);
        
        // Pre-allocate I/O buffers to prevent memory thrashing
        in_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, in_features * sizeof(float), NULL, NULL);
        out_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, out_features * sizeof(float), NULL, NULL);
    }

    ~NativeHoloLayer() {
        clReleaseMemObject(coords_buf);
        clReleaseMemObject(weights_buf);
        clReleaseMemObject(in_buf);
        clReleaseMemObject(out_buf);
    }

    torch::Tensor forward(torch::Tensor input_vec) {
        auto input_contig = input_vec.contiguous();
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input_vec.device());
        auto output_vec = torch::zeros({out_features}, options);

        // Stream 1D input
        clEnqueueWriteBuffer(queue, in_buf, CL_FALSE, 0, in_features * sizeof(float), input_contig.data_ptr(), 0, NULL, NULL);

        // Zero output buffer
        float zero = 0.0f;
        clEnqueueFillBuffer(queue, out_buf, &zero, sizeof(float), 0, out_features * sizeof(float), 0, NULL, NULL);

        // Execute Pure Sparse Math
        clSetKernelArg(spmv_kernel, 0, sizeof(cl_mem), &coords_buf);
        clSetKernelArg(spmv_kernel, 1, sizeof(cl_mem), &weights_buf);
        clSetKernelArg(spmv_kernel, 2, sizeof(cl_mem), &in_buf);
        clSetKernelArg(spmv_kernel, 3, sizeof(cl_mem), &out_buf);
        clSetKernelArg(spmv_kernel, 4, sizeof(int), &num_elements);
        
        size_t global_work = num_elements;
        clEnqueueNDRangeKernel(queue, spmv_kernel, 1, NULL, &global_work, NULL, 0, NULL, NULL);

        // Stream 1D output back
        clEnqueueReadBuffer(queue, out_buf, CL_TRUE, 0, out_features * sizeof(float), output_vec.data_ptr(), 0, NULL, NULL);

        return output_vec;
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init_opencl", &init_opencl, "Initialize Native OpenCL Context");
    pybind11::class_<NativeHoloLayer>(m, "NativeHoloLayer")
        .def(pybind11::init<torch::Tensor, torch::Tensor, int, int, int>())
        .def("forward", &NativeHoloLayer::forward);
}
