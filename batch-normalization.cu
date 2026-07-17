#include <cuda_runtime.h>

void __global__ do_everything(const float* input, const float* gamma, const float* beta, float* output, int N, int C, float eps) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= C) return;
    float mu = 0, sigma = 0, ans = 0;
    float scale = 1.0 / N;
    for (int j = 0; j < N; j++) {
        mu += input[j * C + i];
    }
    mu *= scale;
    for (int j = 0; j < N; j++) {
        sigma += (input[j*C + i]  - mu)*(input[j*C + i]  - mu);
        
    }
    sigma *= scale;
    
    for (int j = 0; j < N; j++) {
        output[j * C + i] = (input[j * C + i] - mu ) / sqrt(sigma + eps) *gamma[i] + beta[i];
    }

}

// input, gamma, beta, output are device pointers
extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output, int N, int C, float eps) {
    int threads = 1024;
    int blocks = ( C + 1023) / 1024;

    do_everything<<<blocks, threads>>>(input, gamma, beta, output, N, C, eps);
    cudaDeviceSynchronize();
}
