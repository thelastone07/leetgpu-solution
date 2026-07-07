#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float tile[256 + 2047];
    int output_size = input_size - kernel_size + 1;

    tile[threadIdx.x] = (x < input_size) ? input[x] : 0.0f;

    for (int i = threadIdx.x; i < kernel_size -1; i+= blockDim.x) {
        int halo_idx = blockDim.x*(blockIdx.x + 1) + i;
        tile[blockDim.x + i] = (halo_idx < input_size) ? input[halo_idx] : 0.0f;
    }

    __syncthreads();

    if (x < output_size) {
        float sum = 0;
        for (int i = 0; i < kernel_size; i++) {
            sum += kernel[i] * tile[threadIdx.x + i];
        }
        output[x] = sum;
    }
                                    
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size,
                      int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size,
                                                              kernel_size);
    cudaDeviceSynchronize();
}
