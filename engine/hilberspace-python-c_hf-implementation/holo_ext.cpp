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

// THE STABLE PHASE-SCALING KERNEL
const char* SPMV_KERNEL_CODE = R"(
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

inline void atomic_add_global_float(volatile __global float *source, const float operand) {
    union { unsigned int int_val; float float_val; } newVal;
    union { unsigned int int_val; float float_val; } prevVal;
    do {
        prevVal.float_val = *source;
        newVal.float_val = prevVal.float_val + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.int_val, newVal.int_val) != prevVal.int_val);
}

__kernel void spmv_hilbert_quantum_batched(
    __global const uint* explicit_rows,   
    __global const uint* explicit_cols, 
    __global const float* weights_real, 
    __global const float* weights_imag, 
    __global const float* input_real,       // Shape: [N, in_features]
    __global float* output_real,            // Shape: [N, out_features]
    __global const float* phase_cosines,    // Shape: [N] (WE PASS COSINES FOR STABILITY)
    const int num_elements,
    const int in_features,
    const int out_features,
    const int N                             // Effective Batch Size (Batch * SeqLen)
) {
    int gid = get_global_id(0);
    int stride = get_global_size(0);

    for (int i = gid; i < num_elements; i += stride) {
        uint row = explicit_rows[i];
        uint col = explicit_cols[i];
        
        float w_r = weights_real[i];
        float w_i = weights_imag[i];
        
        // 1. WAVEFUNCTION COLLAPSE: Reconstruct original amplitude
        float theta = (float)(row + col);
        float w_orig = (w_r * cos(theta)) + (w_i * sin(theta));
        
        // 2. THE PHASE ARRAY LOOP: Apply distinct phase scalars in fast registers
        for (int n = 0; n < N; n++) {
            // Scale the amplitude by the specific phase cosine
            float tuned_w = w_orig * phase_cosines[n];

            int in_idx = (n * in_features) + col;
            int out_idx = (n * out_features) + row;

            float in_r = input_real[in_idx];
            float out_r = tuned_w * in_r;

            if (out_r != 0.0f) {
                atomic_add_global_float(&output_real[out_idx], out_r);
            }
        }
    }
}
)";

void init_opencl() {
    if (is_initialized) return;
    
    // 1. Query the TOTAL number of platforms available
    cl_uint numPlatforms; 
    clGetPlatformIDs(0, NULL, &numPlatforms);
    if (numPlatforms == 0) {
        std::cerr << "[Holoqubed C++] No OpenCL platforms found!\n";
        return;
    }
    
    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), NULL);
    
    num_active_gpus = 0;
    
    // 2. Iterate through ALL platforms to hunt for GPUs
    for (cl_uint p = 0; p < numPlatforms; p++) {
        cl_uint num_devices;
        cl_int err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
        
        if (err != CL_SUCCESS || num_devices == 0) continue;
        
        std::vector<cl_device_id> platform_devices(num_devices);
        clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, num_devices, platform_devices.data(), NULL);
        
        for (cl_uint d = 0; d < num_devices && num_active_gpus < MAX_GPUS; d++) {
            devices[num_active_gpus] = platform_devices[d];
            
            contexts[num_active_gpus] = clCreateContext(NULL, 1, &devices[num_active_gpus], NULL, NULL, NULL);
            
#ifdef CL_VERSION_2_0
            queues[num_active_gpus] = clCreateCommandQueueWithProperties(contexts[num_active_gpus], devices[num_active_gpus], 0, NULL);
#else
            queues[num_active_gpus] = clCreateCommandQueue(contexts[num_active_gpus], devices[num_active_gpus], 0, NULL);
#endif
            // Compile the program specifically for this device's context
            programs[num_active_gpus] = clCreateProgramWithSource(contexts[num_active_gpus], 1, &SPMV_KERNEL_CODE, NULL, NULL);
            clBuildProgram(programs[num_active_gpus], 1, &devices[num_active_gpus], NULL, NULL, NULL);
            
            num_active_gpus++;
        }
    }
    
    is_initialized = true;
    std::cout << "[Holoqubed C++] Multi-GPU Cluster Initialized. Found " << num_active_gpus << " GPUs across " << numPlatforms << " platforms.\n";
}

class NativeHoloLayer {
    cl_mem rows_buf, cols_buf, w_real_buf, w_imag_buf;
    cl_kernel spmv_kernel;
    int in_features, out_features, num_elements, target_gpu;

public:
    NativeHoloLayer(torch::Tensor rows, torch::Tensor cols, torch::Tensor w_real, torch::Tensor w_imag, int in_feat, int out_feat, int num_elem, int gpu_id) {
        in_features = in_feat; 
        out_features = out_feat; 
        num_elements = num_elem;
        
        // Wrap around safely, avoiding modulo by zero if OpenCL initialization failed
        target_gpu = gpu_id % (num_active_gpus > 0 ? num_active_gpus : 1); 
        
        // Allocate buffers specifically in the targeted GPU's VRAM context
        rows_buf = clCreateBuffer(contexts[target_gpu], CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_elements * sizeof(uint32_t), rows.data_ptr(), NULL);
        cols_buf = clCreateBuffer(contexts[target_gpu], CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_elements * sizeof(uint32_t), cols.data_ptr(), NULL);
        w_real_buf = clCreateBuffer(contexts[target_gpu], CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_elements * sizeof(float), w_real.data_ptr(), NULL);
        w_imag_buf = clCreateBuffer(contexts[target_gpu], CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_elements * sizeof(float), w_imag.data_ptr(), NULL);
        
        spmv_kernel = clCreateKernel(programs[target_gpu], "spmv_hilbert_quantum_batched", NULL);
    }

    ~NativeHoloLayer() {
        clReleaseMemObject(rows_buf); 
        clReleaseMemObject(cols_buf);
        clReleaseMemObject(w_real_buf); 
        clReleaseMemObject(w_imag_buf);
        clReleaseKernel(spmv_kernel);
    }

    torch::Tensor forward(torch::Tensor in_real, torch::Tensor phases_cos) {
        int N = in_real.size(0); // This is Batch * SeqLen
        
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(in_real.device());
        auto out_real = torch::zeros({N, out_features}, options);

        cl_mem in_r_buf = clCreateBuffer(contexts[target_gpu], CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * in_features * sizeof(float), in_real.data_ptr(), NULL);
        cl_mem phases_buf = clCreateBuffer(contexts[target_gpu], CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(float), phases_cos.data_ptr(), NULL);
        cl_mem out_r_buf = clCreateBuffer(contexts[target_gpu], CL_MEM_READ_WRITE, N * out_features * sizeof(float), NULL, NULL);

        float zero = 0.0f;
        clEnqueueFillBuffer(queues[target_gpu], out_r_buf, &zero, sizeof(float), 0, N * out_features * sizeof(float), 0, NULL, NULL);

        size_t local_size = 256;
        size_t global_size = 240 * local_size; 
        
        if ((size_t)num_elements < global_size) {
            global_size = (((size_t)num_elements / local_size) + 1) * local_size;
        }

        clSetKernelArg(spmv_kernel, 0, sizeof(cl_mem), &rows_buf);
        clSetKernelArg(spmv_kernel, 1, sizeof(cl_mem), &cols_buf);
        clSetKernelArg(spmv_kernel, 2, sizeof(cl_mem), &w_real_buf);
        clSetKernelArg(spmv_kernel, 3, sizeof(cl_mem), &w_imag_buf);
        clSetKernelArg(spmv_kernel, 4, sizeof(cl_mem), &in_r_buf);
        clSetKernelArg(spmv_kernel, 5, sizeof(cl_mem), &out_r_buf);
        clSetKernelArg(spmv_kernel, 6, sizeof(cl_mem), &phases_buf);
        clSetKernelArg(spmv_kernel, 7, sizeof(int), &num_elements);
        clSetKernelArg(spmv_kernel, 8, sizeof(int), &in_features);
        clSetKernelArg(spmv_kernel, 9, sizeof(int), &out_features);
        clSetKernelArg(spmv_kernel, 10, sizeof(int), &N);

        // GIL ESCAPE HATCH - Allows Python to stream live tokens while GPU crunches math
        {
            pybind11::gil_scoped_release release;
            clEnqueueNDRangeKernel(queues[target_gpu], spmv_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
            clEnqueueReadBuffer(queues[target_gpu], out_r_buf, CL_TRUE, 0, N * out_features * sizeof(float), out_real.data_ptr(), 0, NULL, NULL);
        }

        clReleaseMemObject(in_r_buf);
        clReleaseMemObject(phases_buf);
        clReleaseMemObject(out_r_buf);

        return out_real; 
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init_opencl", &init_opencl, "Initialize Native OpenCL Context");
    pybind11::class_<NativeHoloLayer>(m, "NativeHoloLayer")
        .def(pybind11::init<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int, int>())
        .def("forward", &NativeHoloLayer::forward);
}
