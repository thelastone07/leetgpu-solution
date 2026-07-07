#include <cuda_runtime.h>

__global__ void hist(const int* input, int* histogram, int N, int num_bins) {
    __shared__ int bins[1024];

    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        bins[i] = 0;
    } 
    __syncthreads();
    int st_x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (st_x  < N ) {
        atomicAdd(&bins[input[ st_x]],1);
    }

    __syncthreads();

    if (threadIdx.x < num_bins)
    atomicAdd(&histogram[threadIdx.x], bins[threadIdx.x]);
}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + 1024 - 1) / (1024);

    cudaMemset(histogram, 0, sizeof(int) * num_bins);
    hist<<<blocksPerGrid, threadsPerBlock>>>(input, histogram, N, num_bins);
    cudaDeviceSynchronize();
}
