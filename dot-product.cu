#include <cuda_runtime.h>

__global__ void firstWave(const float* A, const float* B, int N, float* sums) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;

    float result = 0;
    int v4_limit = N/4;
    if (x < v4_limit) {
        float4 a4 = reinterpret_cast<const float4*>(A)[x];
        float4 b4 = reinterpret_cast<const float4*>(B)[x];
        
        result += a4.x * b4.x + a4.y * b4.y + a4.z * b4.z + a4.w * b4.w;
    }

    for (int i = v4_limit * 4 + threadIdx.x; i < N; i += blockDim.x*gridDim.x)
        result += A[i] * B[i];
    
    int lane = threadIdx.x % 32;

    for (int i = 16; i >= 1; i /= 2) {
        float other_result = __shfl_down_sync(0xffffffff, result, i);
        if (lane < i) 
            result += other_result;
    }
    __shared__ float t[32];
    if (lane == 0) t[threadIdx.x / 32] = result;

    __syncthreads();

    for (int i = 16; i >= 1; i /= 2) {
        if (threadIdx.x < i) {
            t[threadIdx.x] += t[threadIdx.x + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    sums[blockIdx.x] = t[0];
}

__global__ void secondWave(float* input, float* sums, int N) {
    int x = blockDim.x * blockIdx.x + threadIdx.x ;

    float result = 0;
    int v4_limit = N / 4;
    if (x < v4_limit) {
        float4 a4 = reinterpret_cast<const float4*>(input)[x];
        result += a4.x + a4.y + a4.z + a4.w;
    }
    for (int i = v4_limit * 4 + threadIdx.x; i < N; i += blockDim.x*gridDim.x) result += input[i];

    int lane = threadIdx.x % 32;
    for (int i = 16; i >= 1; i /= 2) {
        float other_result = __shfl_down_sync(0xffffffff, result, i);
        if (lane < i) result += other_result;
    }

    __shared__ float t[32];
    if (lane == 0) t[threadIdx.x / 32] = result;

    __syncthreads();

    for (int i = 16; i >= 1; i /= 2) {
        if (threadIdx.x < i) {
            t[threadIdx.x] += t[threadIdx.x + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) sums[blockIdx.x] = t[0];
}



extern "C" void solve(const float* A, const float* B, float* result, int N) {
    int threads = 1024;
    int blocks = (N + threads*4 - 1) /( threads * 4);

    float* sums;
    cudaMalloc(&sums, sizeof(float) * blocks);
    firstWave<<< blocks, threads >>> (A, B, N, sums);
    cudaDeviceSynchronize();

    int blocks_2 = (blocks + threads*4 -1) / (threads* 4);
    float* sec_sums;
    cudaMalloc(&sec_sums, sizeof(float) * blocks_2);
    secondWave<<<blocks_2, threads>>>(sums, sec_sums, blocks);
    cudaDeviceSynchronize();

    int blocks_3 = (blocks_2 + threads* 4 - 1) / (threads * 4);
    secondWave<<<blocks_3, threads>>>(sec_sums, result, blocks_2);
    cudaDeviceSynchronize();

    cudaFree(sums);
    cudaFree(sec_sums);
}
