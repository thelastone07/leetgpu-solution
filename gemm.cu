#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda::wmma;

template< int BM, int BN, int BK>
__global__ void gemm(const half* A, const half* B, half* C, int M , int N, int K, float alpha, float beta) {
    __shared__ half a[BM][BK], b[BK][BN];
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    int numThreads = blockDim.x * blockDim.y;

    int warp_id = tid / 32;
    int r = warp_id / 4; // 0..1
    int c = warp_id % 4; // 0..3

    fragment<accumulator, 16, 16, 16, float> acc[4][2];
    for (int m = 0; m < 4; m++)
    for (int n = 0; n < 2; n++) fill_fragment(acc[m][n], 0.0f); 

    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;

    for (int bk = 0; bk < (K + BK -1) / BK; bk++) { // (K + BK - 1) / BK how many strides along K direction ceil div
        for (int i = 0; i < BM * BK; i += numThreads) {
            int idx = tid + i;
            int r = idx / BK;
            int c = idx % BK;
            int st_r = blockIdx.y * BM;
            int st_c = bk * BK;
            a[r][c] = (st_r + r < M && st_c + c < K) ? A[(st_r + r) * K + st_c + c] : __float2half(0);
        }
        
        for (int i = 0; i < BN * BK; i += numThreads) {
            int idx = tid + i ;
            int r = idx / BN;
            int c = idx % BN;
            int st_c = blockIdx.x * BN ;
            int st_r = bk * BK;
            b[r][c] = (st_r + r < K && st_c + c < N) ? B[(st_r + r)* N + st_c + c] : __float2half(0);
        }
        __syncthreads();

        for (int k = 0; k < BK; k+=16) {
            for (int m = 0; m < 4; m++) {
                load_matrix_sync(a_frag, &a[r * 64 + m * 16][k], BK);
                for (int n = 0; n < 2; n++) {
                    load_matrix_sync(b_frag, &b[k][c * 32 + n * 16], BN);
                    mma_sync(acc[m][n], a_frag, b_frag, acc[m][n]);
                }
            }
        }
        __syncthreads();
    }
    
    for (int m = 0; m < 4; m++) {
        for (int n = 0; n < 2; n++) {
            int c_row = blockIdx.y * BM + r * 64 +  m * 16;
            int c_col = blockIdx.x * BN + c * 32 + n * 16;
            fragment<accumulator, 16, 16, 16, half> c_frag;
            if (c_row + 15 < M && c_col + 15 < N) {
                load_matrix_sync(c_frag, &C[c_row * N + c_col], N, mem_row_major);
                for (int i = 0; i < c_frag.num_elements; i++) {
                    c_frag.x[i] = __float2half(alpha * acc[m][n].x[i] + beta * __half2float(c_frag.x[i]));
                }
                store_matrix_sync(&C[c_row * N + c_col], c_frag, N, mem_row_major);
            }
        }
    }

}

__global__ void static_gemm(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    const int TILE = 32;
    __shared__ float a[TILE][TILE], b[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0;
    for (int k = 0; k < (K + 31) / 32; k++) {
        int ak = TILE * k + threadIdx.x;
        int bk = TILE * k + threadIdx.y;

        a[threadIdx.y][threadIdx.x] = (ak < K && row < M) ? __half2float(A[row * K + ak]) : 0.0f;
        b[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? __half2float(B[bk * N + col]) : 0.0f;
        __syncthreads();

        for (int kk = 0; kk < 32; kk++) {
            sum += a[threadIdx.y][kk] * b[kk][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = __float2half(alpha * sum + beta * __half2float(C[row * N + col]));
}

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    constexpr int BM = 128, BN = 128, BK = 32;
    if (N % 16 == 0 && M % 16 == 0 && K % 16 == 0) {
        dim3 threadsPerBlock = dim3(BN / 8, BM / 8);
        dim3 blocksPerGrid = dim3((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm<BM, BN, BK><<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);
    }
    else {
        dim3 threadsPerBlock = dim3(32, 32);
        dim3 blocksPerGrid = dim3( (N + 31) / 32, (M + 31) / 32);
        static_gemm<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);
    }
    cudaDeviceSynchronize();
}
