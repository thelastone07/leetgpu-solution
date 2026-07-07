#include <cuda_runtime.h>

__global__ void block_sum(const float* input, float* output, int N, float* block_sums ){

}

__global__ void reduce_offsets(float* output, int len) {

}

__global__ void local_scan(float* block_sums, )

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threads = 1024;
    int blocks = (N + threads - 1) / threads;

    float* block_sums;
    cudaMalloc(&block_sums, sizeof(float) * blocks);
    block_sum<<<blocks, threads>>>(input, output, N, block_sums);
    cudaDeviceSynchronize();

    float* reduced_sums;
    int blocks1 = (blocks + threads - 1) / threads;
    cudaMalloc(&reduced_sums, sizeof(float) * blocks1);
    reduce_offsets<<<blocks1, threads>>>(reduced_sums, blocks);
    cudaDeviceSynchronize();

    
}
