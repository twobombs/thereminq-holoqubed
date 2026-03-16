#include <torch/extension.h>
#include <CL/cl.h>
#include <iostream>
#include <vector>

cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel scatter_kernel;
cl_kernel gemv_kernel;
bool is_initialized = false;

const char* KERNEL_CODE = R"(
__kernel void scatter_weights(__global const ulong* morton_coords, __global const ushort* sparse_weights, __global float* dense_matrix, const int num_elements, const int in_features) {
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

    dense_matrix[row * in_features + col] = weight;
}

__kernel void gemv_math(__global const float* dense_matrix, __global const float* input_vector, __global float* output_vector, const int in_features, const int out_features) {
    int row = get_global_id(0);
    if (row >= out_features) return;

    float sum = 0.0f;
    int row_offset = row * in_features;
    for (int col = 0; col < in_features; col++) {
        sum += dense_matrix[row_offset + col] * input_vector[col];
    }
    output_vector[row] = sum;
}
)";

void init_opencl() {
    if (is_initialized) return;
    cl_uint numPlatforms; clGetPlatformIDs(1, NULL, &numPlatforms);
    cl_platform_id platform; clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device; clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device, 0, NULL);
    program = clCreateProgramWithSource(context, 1, &KERNEL_CODE, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    
    scatter_kernel = clCreateKernel(program, "scatter_weights", NULL);
    gemv_kernel = clCreateKernel(program, "gemv_math", NULL);

    is_initialized = true;
    std::cout << "[Holoqubed C++] Native OpenCL Pre-Inflated Engine Initialized.\n";
}

class NativeHoloLayer {
    cl_mem dense_buf;
    cl_mem in_buf;
    cl_mem out_buf;
    int in_features;
    int out_features;

public:
    NativeHoloLayer(torch::Tensor coords, torch::Tensor weights, int in_feat, int out_feat, int num_elem) {
        in_features = in_feat;
        out_features = out_feat;
        size_t dense_bytes = out_features * in_features * sizeof(float);

        // 1. Allocate Temp Sparse Buffers
        cl_mem coords_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_elem * sizeof(uint64_t), coords.data_ptr(), NULL);
        cl_mem weights_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_elem * sizeof(uint16_t), weights.data_ptr(), NULL);
        
        // 2. Allocate Permanent Dense Buffer
        dense_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, dense_bytes, NULL, NULL);
        float zero = 0.0f;
        clEnqueueFillBuffer(queue, dense_buf, &zero, sizeof(float), 0, dense_bytes, 0, NULL, NULL);

        // 3. Scatter Once During Boot
        clSetKernelArg(scatter_kernel, 0, sizeof(cl_mem), &coords_buf);
        clSetKernelArg(scatter_kernel, 1, sizeof(cl_mem), &weights_buf);
        clSetKernelArg(scatter_kernel, 2, sizeof(cl_mem), &dense_buf);
        clSetKernelArg(scatter_kernel, 3, sizeof(int), &num_elem);
        clSetKernelArg(scatter_kernel, 4, sizeof(int), &in_features);
        size_t global_scatter = num_elem;
        clEnqueueNDRangeKernel(queue, scatter_kernel, 1, NULL, &global_scatter, NULL, 0, NULL, NULL);
        clFinish(queue); // Ensure scatter completes before destroying temp buffers

        // 4. Destroy Sparse Buffers (Free VRAM)
        clReleaseMemObject(coords_buf);
        clReleaseMemObject(weights_buf);

        // 5. Pre-allocate I/O buffers for generation
        in_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, in_features * sizeof(float), NULL, NULL);
        out_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, out_features * sizeof(float), NULL, NULL);
    }

    ~NativeHoloLayer() {
        clReleaseMemObject(dense_buf);
        clReleaseMemObject(in_buf);
        clReleaseMemObject(out_buf);
    }

    torch::Tensor forward(torch::Tensor input_vec) {
        auto input_contig = input_vec.contiguous();
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input_vec.device());
        auto output_vec = torch::zeros({out_features}, options);

        // Stream 1D input
        clEnqueueWriteBuffer(queue, in_buf, CL_FALSE, 0, in_features * sizeof(float), input_contig.data_ptr(), 0, NULL, NULL);

        // Fast OpenCL Dense Math
        clSetKernelArg(gemv_kernel, 0, sizeof(cl_mem), &dense_buf);
        clSetKernelArg(gemv_kernel, 1, sizeof(cl_mem), &in_buf);
        clSetKernelArg(gemv_kernel, 2, sizeof(cl_mem), &out_buf);
        clSetKernelArg(gemv_kernel, 3, sizeof(int), &in_features);
        clSetKernelArg(gemv_kernel, 4, sizeof(int), &out_features);
        size_t global_gemv = out_features;
        clEnqueueNDRangeKernel(queue, gemv_kernel, 1, NULL, &global_gemv, NULL, 0, NULL, NULL);

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
