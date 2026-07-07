#include <cuda_runtime.h>

__global__ void matmul(const float* A, const float* B, float* C, int BATCH, int M, int N, int K)  {
    const int TILE = 32;
    __shared__ float a[32][32], b[32][32];
    int bid = blockIdx.z;
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0;

    if (bid >= BATCH) return;

    for (int k = 0; k < (K + 31) / 32; k++) {
        int ak = TILE * k + threadIdx.x;
        int bk = TILE * k + threadIdx.y;

        a[threadIdx.y][threadIdx.x] = (ak < K && row < M) ? A[bid * M * K + row * K + ak] : 0.0f;
        b[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? B[bid * K * N + bk * N + col] : 0.0f;
        __syncthreads();

        for (int kk = 0; kk < 32; kk++) {
            sum += a[threadIdx.y][kk] * b[kk][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[bid * M * N + row * N + col] = sum; 
    }
    
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    dim3 threads(32, 32);
    dim3 blocks((N+31)/32, (M+31)/32, BATCH);
    matmul<<<blocks, threads>>>(A, B, C, BATCH, M, N, K);
    cudaDeviceSynchronize();
}
    