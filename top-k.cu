#include <cuda_runtime.h>
#include <float.h>

__device__ void swap(float *a, float* b) {
    float tmp = *a;
    *a = *b;
    *b = tmp;
}

__global__ void global_bitonic(float* input, int N, int k_step, int stride) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int partner = i ^ stride;
    if (partner < i || i >= N) return;
    
    float* val1 = &input[i], *val2 = &input[partner];
    if ((i & k_step) != 0) {
        if (*val1 > *val2) swap(val1, val2);
    } else {
        if (*val1 < *val2) swap(val1, val2);
    }
}

__global__ void shared_bitonic(float* input, int N, int k_step) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    __shared__ float temp[1024];
    temp[tid] = (i < N)  ? input[i] : -FLT_MAX;
    __syncthreads();
    for (int stride = min(k_step / 2, blockDim.x / 2); stride >= 1; stride /= 2) {
        int partner = tid ^ stride;
        if (partner > tid) { 
            float* val1 = &temp[tid], *val2 = &temp[partner];
            if ((i & k_step) != 0) {
                if (*val1 > *val2) swap(val1, val2);
            } else {
                if (*val1 < *val2) swap(val1, val2);
            }
        }
        __syncthreads();
    }
    input[i] = temp[tid];
}

__global__ void fill(float *temp, int N, int modified_N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= modified_N - N) return;
    temp[tid + N] = -FLT_MAX;
}

__global__ void fill_output(float* temp, float* output, int k) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= k) return;
    output[i] = temp[i];
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int k) {
    int modified_N = 1 << (int)ceil(log2(N));
    
    float * temp;
    cudaMalloc(&temp, sizeof(float) * modified_N);
    cudaMemcpy(temp, input, sizeof(float) * N, cudaMemcpyDeviceToDevice);

    if (modified_N > N) {
        int threads = 1024;
        int blocks = (modified_N - N + 1023) / 1024;
        fill<<<blocks, threads>>>(temp, N, modified_N);
        cudaDeviceSynchronize();
    }

    int threads = 1024;
    int blocks = (modified_N + 1023) / 1024;
    for (int k_step = 2; k_step <= modified_N; k_step *= 2) {
        if (k_step <= 512) {shared_bitonic<<<blocks, threads>>>(temp, modified_N, k_step); cudaDeviceSynchronize(); }
        else for(int stride = k_step / 2; stride >= 1; stride /= 2) {
            if (stride <= 512){ shared_bitonic<<<blocks, threads>>>(temp, modified_N, k_step); cudaDeviceSynchronize(); break;}
            else global_bitonic<<<blocks, threads>>>(temp, modified_N, k_step, stride);
            cudaDeviceSynchronize();
        }
    }
    
    blocks = (k + 1023) / 1024;
    fill_output<<<blocks, threads>>>(temp, output, k);
    cudaDeviceSynchronize();

    cudaFree(temp);
}
