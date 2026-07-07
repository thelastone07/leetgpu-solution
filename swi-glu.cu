#include <cuda_runtime.h>

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x < halfN) {
        float x1 = input[x];
        float x2 = input[x + halfN];
        float silu = x1 / (1.0f + expf(-x1));
        output[x] = silu * x2;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
