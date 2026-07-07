#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float tileA[16][16];
    __shared__ float tileB[16][16];

    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;


    int i = threadIdx.y;
    int j = threadIdx.x;
    float sum = 0.0;
    for (int t = 0; t < (N+15)/16; t++) {
        tileA[i][j] = (t * 16 + j < N && row < M) ? A[row * N + (t*16+j)] : 0.0f;
        tileB[i][j] = (t * 16 + i < N && col < K) ? B[(t*16+i)* K + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < 16; k++) {
            sum += tileA[i][k] * tileB[k][j];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
