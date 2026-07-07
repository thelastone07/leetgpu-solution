#include <cuda_runtime.h>

__global__ void mse_kernel(const float* a, const float* b, float* sums, int N) {
    float sum = 0;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }

    for (int i = 16; i >= 1; i /= 2) sum += __shfl_down_sync(0xffffffff, sum, i);

    __shared__ float wsum[32];
    int tid = threadIdx.x;
    if (tid % 32 == 0) wsum[tid / 32] = sum;
    __syncthreads();

    if (tid / 32 == 0) {
        sum = wsum[tid % 32];
        for (int i = 16; i >= 1; i /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, i);
        }
    }


    if (tid == 0)
    sums[blockIdx.x] = sum;
}

__global__ void reduce(float *sums, float* mse, int N) {
    int tid = threadIdx.x;
    float sum = tid < N ? sums[tid] : 0;
    for (int i = 16; i >= 1; i /= 2) sum += __shfl_down_sync(0xffffffff, sum, i);

    __shared__ float wsum[32];
    if (tid % 32 == 0) wsum[tid / 32] = sum;
    __syncthreads();

    if (tid / 32 == 0) {
        sum = wsum[tid % 32];
        for (int i = 16; i >= 1; i /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, i);
        }
    }

    if (tid == 0) *mse = sum / N;
}

// predictions, targets, mse are device pointers
extern "C" void solve(const float* predictions, const float* targets, float* mse, int N) {
    int threads = 1024;
    int blocks = min(1024, (N + 1023) / 1024);
    float *sums;
    cudaMalloc(&sums, sizeof(float) * blocks);
    mse_kernel<<<blocks, threads>>>(predictions, targets, sums, N);
    cudaDeviceSynchronize();
    reduce<<<1,1024>>>(sums, mse, blocks);
    cudaDeviceSynchronize();
    cudaFree(sums);
}
