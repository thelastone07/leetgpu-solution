#include <cuda_runtime.h>

__global__ void multiply(const float* A, const float* x, float* y, int M , int N) {

    float sum = 0;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        if (A[blockIdx.x * N + i] == 0) continue;
        sum += A[blockIdx.x * N + i] * x[i];
    }
    
    int lane = threadIdx.x % 32;
    for (int i = 16; i >= 1; i /= 2) {
        float other_sum = __shfl_down_sync(0xffffffff, sum, i);
        if (lane < i)
        sum += other_sum;
    }
    __shared__ float t[32];
    if (lane == 0) t[threadIdx.x / 32] = sum;
    __syncthreads();
    for (int i = 16; i >= 1; i /= 2) {
        if (threadIdx.x < i) {
            t[threadIdx.x] += t[threadIdx.x + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) y[blockIdx.x] = t[0];    
}


// A, x, y are device pointers
extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {
    int threads = 1024;
    int blocks = M; 
    multiply<<<blocks, threads>>>(A, x, y, M , N);
    cudaDeviceSynchronize();
}
