#include <cuda_runtime.h>

__global__ void matmul(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K, float scale_A, float scale_b, float scale_c, int zero_point_A, int zero_point_B, int zero_point_C)  {
    const int TILE = 32;
    __shared__ int8_t a[32][32], b[32][32];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    int32_t sum = 0;

    for (int k = 0; k < (K + 31) / 32; k++) {
        int ak = TILE * k + threadIdx.x;
        int bk = TILE * k + threadIdx.y;

        a[threadIdx.y][threadIdx.x] = (ak < K && row < M) ?A[ row * K + ak] : (int8_t) zero_point_A;
        b[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? B[ bk * N + col] : (int8_t)zero_point_B;
        __syncthreads();

        for (int kk = 0; kk < 32; kk++) {
            sum += (int32_t)(a[threadIdx.y][kk] -zero_point_A)* (int32_t)(b[kk][threadIdx.x]- zero_point_B);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        float result = (float) sum * scale_A * scale_b / scale_c;
        int iresult = (int)result + ((result - (int)result) > 0.5f ? 1 : 0) + zero_point_C;
        C[row * N + col] = (int8_t)max(-128, min(127, iresult));
    }
    
}

// A, B, C are device pointers
extern "C" void solve(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K, float scale_A, float scale_B, float scale_C, int zero_point_A, int zero_point_B, int zero_point_C) {
    dim3 threads(32,32);
    dim3 blocks( (N + 31) / 32, (M + 31) / 32);
    matmul<<<blocks, threads>>>(A, B, C, M, N, K, scale_A, scale_B, scale_C, zero_point_A, zero_point_B, zero_point_C);
    cudaDeviceSynchronize();
}
