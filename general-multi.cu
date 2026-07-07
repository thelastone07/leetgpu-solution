#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda::wmma;

template<int BM = 128, int BN = 128, int BK = 32, int TM = 8, int TN = 8>
__global__ void gmem(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {

    __shared__ half a[BM][BK], b[BK][BN];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * blockDim.y;

    int warp_id = tid / 32; // max tid 
    int warp_row = warp_id / 2; // 4 along M
    int warp_col = warp_id % 2; // 2 along N 

    fragment<accumulator, 16, 16, 16, float> acc[2][4];
    for (int m = 0; m < 2; m++)
    for (int n = 0; n < 4; n++) fill_fragment(acc[m][n], 0.0f);
    
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, row_major> b_frag; 

    for(int bk = 0; bk < (K + BK - 1) / BK; bk++){   
        //outer loop for adding 128x 32 into a and b. = 4096 elements 
        for (int i = 0; i < BM * BK / numThreads; i++) {
            int idx = tid + i * numThreads;
            int row = idx / BK;
            int col = idx % BK;
            a[row][col] = (blockIdx.y * BM + row < M && bk * BK + col < K) 
                ? A[(blockIdx.y * BM + row) * K + bk * BK + col] : __float2half(0.f);
        }

        for (int i = 0; i < BN * BK / numThreads; i++) {
            int idx = tid + i * numThreads;
            int row = idx / BN;
            int col = idx % BN;
            b[row][col] = (bk * BK + row < K && blockIdx.x * BN + col < N)
                ? B[(bk * BK + row) * N + blockIdx.x * BN + col] : __float2half(0.f);
        }

        __syncthreads();

        for (int k = 0; k < BK; k += 16) {
            for (int m = 0; m < 2; m++) {
                load_matrix_sync(a_frag, &a[warp_row*32 + m*16][k], BK);
                for (int n = 0; n < 4; n++) {
                    load_matrix_sync(b_frag, &b[k][warp_col*64 + n*16], BN);
                    mma_sync(acc[m][n], a_frag, b_frag, acc[m][n]);
                }
            }
        }

        __syncthreads();
    }

    for (int m = 0; m < 2; m++) {
        for (int n = 0; n < 4; n++) {
            int c_row = blockIdx.y * BM + warp_row * 32 + m * 16;
            int c_col = blockIdx.x * BN + warp_col * 64 + n * 16;

            fragment<accumulator, 16, 16, 16, half> c_frag;
            if (c_row + 15 < M && c_col + 15 < N) {
                load_matrix_sync(c_frag, &C[c_row * N + c_col], N, mem_row_major);
                for (int i = 0; i < c_frag.num_elements; i++)
                    c_frag.x[i] = __float2half(alpha * acc[m][n].x[i] + beta * __half2float(c_frag.x[i]));
                store_matrix_sync(&C[c_row * N + c_col], c_frag, N, mem_row_major);
            }
        }
    }
}

__global__ void scalar_gemm(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    const int TILE = 32;
    __shared__ float sa[TILE][TILE], sb[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.f;
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int ak = t * TILE + threadIdx.x;
        int bk = t * TILE + threadIdx.y;
        sa[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? __half2float(A[row * K + ak]) : 0.f;
        sb[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? __half2float(B[bk * N + col]) : 0.f;
        __syncthreads();
        for (int k = 0; k < TILE; k++) sum += sa[threadIdx.y][k] * sb[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = __float2half(alpha * sum + beta * __half2float(C[row * N + col]));
}

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    constexpr int BM = 128, BN = 128, BK = 32, TM = 8, TN = 8;

    if (M % 16 == 0 && N % 16 == 0 && K % 16 == 0) {
        dim3 threads(BN / TN, BM / TM);
        dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
        gmem<BM, BN, BK, TM, TN><<<blocks, threads>>>(A, B, C, M, N, K, alpha, beta);
    } else {
        dim3 threads(32, 32);
        dim3 blocks((N + 31) / 32, (M + 31) / 32);
        scalar_gemm<<<blocks, threads>>>(A, B, C, M, N, K, alpha, beta);
    }
    cudaDeviceSynchronize();
}
