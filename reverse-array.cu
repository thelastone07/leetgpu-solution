#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < N / 2) {
        float temp = input[x];
        input[x] = input[N - x - 1];
        input[N - x - 1] = temp;
    }
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
