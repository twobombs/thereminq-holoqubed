#include <torch/extension.h>
#include <CL/cl.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <pybind11/pybind11.h>

#define MAX_GPUS 8
cl_context contexts[MAX_GPUS];
cl_command_queue queues[MAX_GPUS];
cl_device_id devices[MAX_GPUS];
cl_program programs[MAX_GPUS];
int num_active_gpus = 0;
bool is_initialized = false;

// THE JIT REHYDRATOR KERNEL (ZERO ATOMICS REQUIRED)
const char* JIT_DECOMPRESS_KERNEL = R"(
__kernel void rehydrate_quantum_weights(
    __global const uint* explicit_rows,   
    __global const uint* explicit_cols, 
    __global const float* weights_real, 
    __global const float* weights_imag, 
    __global float* output_dense_matrices,  // Shape: [N, out_features, in_features]
    __global const float* phase_offsets_rad, 
    const int num_elements,
    const int in_features,
    const int out_features,
    const int N                             
) {
    int gid = get_global_id(0);
    int stride = get_global_size(0);

    for (int i = gid; i < num_elements; i += stride) {
        uint row = explicit_rows[i];
        uint col = explicit_cols[i];
        
        float w_r = weights_real[i];
        float w_i = weights_imag[i];
        
        // 1. COLLAPSE: Reconstruct original signed amplitude
        float base_theta = (float)(row + col);
        float A = (w_r * cos(base_theta)) + (w_i * sin(base_theta));
        
        // 2. MODULATE: True spatial frequency interference
        for (int n = 0; n < N; n++) {
            float tuned_w = A * cos(phase_offsets_rad[n] * base_theta);

            // Write EXACTLY ONCE to the pre-zeroed dense matrix. No Atomics needed!
            // Memory layout: [Batch N][Row][Col]
            int flat_idx = (n * out_features * in_features) + (row * in_features) + col;
            output_dense_matrices[flat_idx] = tuned_w;
        }
    }
}
)";

void init_opencl() {
    if (is_initialized) return;
    
    cl_uint num_platforms;
    clGetPlatformIDs(0, NULL, &num_platforms);
    if (num_platforms == 0) {
        std::cerr << "[Holoqubed Error] No OpenCL platforms found!\n";
        return;
    }
    
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), NULL);
    
    num_active_gpus = 0;
    
    for (cl_uint p = 0; p < num_platforms; ++p) {
        cl_uint num_devices = 0;
        if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices) != CL_SUCCESS || num_devices == 0) continue; 
        
        std::vector<cl_device_id> plat_devices(num_devices);
        clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, num_devices, plat_devices.data(), NULL);
        
        for (cl_uint d = 0; d < num_devices; ++d) {
            if (num_active_gpus < MAX_GPUS) {
                devices[num_active_gpus] = plat_devices[d];
                
                char deviceName[256];
                clGetDeviceInfo(devices[num_active_gpus], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
                std::cout << "[Holoqubed C++] Discovered GPU: " << deviceName << "\n";
                
                cl_context_properties props[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[p], 0 };
                contexts[num_active_gpus] = clCreateContext(props, 1, &devices[num_active_gpus], NULL, NULL, NULL);
                
#ifdef CL_VERSION_2_0
                queues[num_active_gpus] = clCreateCommandQueueWithProperties(contexts[num_active_gpus], devices[num_active_gpus], 0, NULL);
#else
                queues[num_active_gpus] = clCreateCommandQueue(contexts[num_active_gpus], devices[num_active_gpus], 0, NULL);
#endif
                
                programs[num_active_gpus] = clCreateProgramWithSource(contexts[num_active_gpus], 1, &JIT_DECOMPRESS_KERNEL, NULL, NULL);
                clBuildProgram(programs[num_active_gpus], 1, &devices[num_active_gpus], NULL, NULL, NULL);
                
                num_active_gpus++;
            }
        }
    }
    
    is_initialized = true;
    std::cout << "[Holoqubed C++] Multi-GPU Cluster Initialized. Bound to " << num_active_gpus << " GPUs.\n";
}

class NativeHoloLayer {
    cl_mem rows_buf, cols_buf, w_real_buf, w_imag_buf;
    cl_kernel decompress_kernel;
    int in_features, out_features, num_elements, target_gpu;

public:
    NativeHoloLayer(torch::Tensor rows, torch::Tensor cols, torch::Tensor w_real, torch::Tensor w_imag, int in_feat, int out_feat, int num_elem, int gpu_id) {
        in_features = in_feat; 
        out_features = out_feat; 
        num_elements = num_elem;
        target_gpu = gpu_id % (num_active_gpus > 0 ? num_active_gpus : 1); 
        
        rows_buf = clCreateBuffer(contexts[target_gpu], CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_elements * sizeof(uint32_t), rows.data_ptr(), NULL);
        cols_buf = clCreateBuffer(contexts[target_gpu], CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_elements * sizeof(uint32_t), cols.data_ptr(), NULL);
        w_real_buf = clCreateBuffer(contexts[target_gpu], CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_elements * sizeof(float), w_real.data_ptr(), NULL);
        w_imag_buf = clCreateBuffer(contexts[target_gpu], CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_elements * sizeof(float), w_imag.data_ptr(), NULL);
        
        decompress_kernel = clCreateKernel(programs[target_gpu], "rehydrate_quantum_weights", NULL);
    }

    ~NativeHoloLayer() {
        clReleaseMemObject(rows_buf); 
        clReleaseMemObject(cols_buf);
        clReleaseMemObject(w_real_buf); 
        clReleaseMemObject(w_imag_buf);
        clReleaseKernel(decompress_kernel);
    }

    // Now returns the fully instantiated dense weights, rather than the final multiplied vector
    torch::Tensor get_dense_weights(torch::Tensor phases_rad) {
        int N = phases_rad.size(0); 
        
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(phases_rad.device());
        auto out_dense_matrices = torch::zeros({N, out_features, in_features}, options);

        cl_mem phases_buf = clCreateBuffer(contexts[target_gpu], CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(float), phases_rad.data_ptr(), NULL);
        cl_mem out_mats_buf = clCreateBuffer(contexts[target_gpu], CL_MEM_READ_WRITE, N * out_features * in_features * sizeof(float), NULL, NULL);

        float zero = 0.0f;
        clEnqueueFillBuffer(queues[target_gpu], out_mats_buf, &zero, sizeof(float), 0, N * out_features * in_features * sizeof(float), 0, NULL, NULL);

        size_t local_size = 256;
        size_t global_size = 240 * local_size; 
        if ((size_t)num_elements < global_size) {
            global_size = (((size_t)num_elements / local_size) + 1) * local_size;
        }

        clSetKernelArg(decompress_kernel, 0, sizeof(cl_mem), &rows_buf);
        clSetKernelArg(decompress_kernel, 1, sizeof(cl_mem), &cols_buf);
        clSetKernelArg(decompress_kernel, 2, sizeof(cl_mem), &w_real_buf);
        clSetKernelArg(decompress_kernel, 3, sizeof(cl_mem), &w_imag_buf);
        clSetKernelArg(decompress_kernel, 4, sizeof(cl_mem), &out_mats_buf);
        clSetKernelArg(decompress_kernel, 5, sizeof(cl_mem), &phases_buf);
        clSetKernelArg(decompress_kernel, 6, sizeof(int), &num_elements);
        clSetKernelArg(decompress_kernel, 7, sizeof(int), &in_features);
        clSetKernelArg(decompress_kernel, 8, sizeof(int), &out_features);
        clSetKernelArg(decompress_kernel, 9, sizeof(int), &N);

        {
            pybind11::gil_scoped_release release;
            clEnqueueNDRangeKernel(queues[target_gpu], decompress_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
            clEnqueueReadBuffer(queues[target_gpu], out_mats_buf, CL_TRUE, 0, N * out_features * in_features * sizeof(float), out_dense_matrices.data_ptr(), 0, NULL, NULL);
        }

        clReleaseMemObject(phases_buf);
        clReleaseMemObject(out_mats_buf);

        return out_dense_matrices; 
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init_opencl", &init_opencl, "Initialize Native OpenCL Context");
    pybind11::class_<NativeHoloLayer>(m, "NativeHoloLayer")
        .def(pybind11::init<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int, int>())
        .def("get_dense_weights", &NativeHoloLayer::get_dense_weights);
}
