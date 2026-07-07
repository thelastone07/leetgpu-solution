#include <cuda_runtime.h>

__global__ void reduction(const float* input, float* output , int N) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float temp[256];
    temp[threadIdx.x] = (x < N) ? input[x] : 0.0f;
    __syncthreads();
    for (int i = blockDim.x / 2; i >= 1; i = i / 2) {
        if (threadIdx.x < i) {
            temp[threadIdx.x] += temp[threadIdx.x + i];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) 
        atomicAdd(output, temp[0]);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock -1) / threadsPerBlock;
    cudaMemset(output, 0, sizeof(float));
    reduction<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
