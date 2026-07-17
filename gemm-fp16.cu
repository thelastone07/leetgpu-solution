#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include<mma.h>
using namespace nvcuda;

const int BK = 64, BM = 128, BN = 64;


// A, B, C are device pointers
__global__ void matmul(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) {
    int batch = blockIdx.z;
    if (batch >= BATCH) return;
    int idx = threadIdx.y * blockDim.x + threadIdx.x;
    int st_r = blockIdx.y * BM;
    int st_c = blockIdx.x * BN;
    __shared__ half a[BM][BK], b[BK][BN];

    int warp = threadIdx.y;    
    int WM = BM / 16, WN = BN / 16; // 8 x 4
    int r = warp % 8;
    int c = warp / 8;


    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);


    for (int k = 0; k < K; k += BK) {
        for (int i = idx ; i < BM * BK; i += blockDim.x * blockDim.y) {
            int y = i / BK;
            int x = i % BK;
            int r = st_r + y;
            int c = x + k;
            a[y][x] = (r < M &&  c < K) ? A[batch * M * K + r * K + c ] : __float2half(0.0f);
        }

        for (int i = idx; i < BN * BK; i += blockDim.x * blockDim.y) {
            int y = i / BN;
            int x = i % BN;
            int r = y + k;
            int c = st_c + x;
            b[y][x] = (r < K && c < N) ? B[batch * N * K + r * N + c] : __float2half(0.0f);
        }

        __syncthreads();

        for (int dk = 0; dk < BK; dk += 16) {
            wmma::load_matrix_sync(a_frag, &a[r * 16][dk], BK);
            wmma::load_matrix_sync(b_frag, &b[dk][c * 16], BN);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        __syncthreads();
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag_half;
    for (int i = 0; i < c_frag.num_elements; i++) {
        c_frag_half.x[i] = __float2half(c_frag.x[i]);
    }
    __shared__  half c_half[BM][BN];
    wmma::store_matrix_sync(&c_half[r*16][c*16], c_frag_half, BN, wmma::mem_row_major);
    __syncthreads();

    for (int i = idx; i < BM * BN; i += blockDim.x * blockDim.y) {
        int y = i / BN;
        int x = i % BN;
        int row = st_r + y;
        int col = st_c + x;
        if (row < M && col < N) {
            C[batch * M * N + row * N + col] = c_half[y][x];
        }
    }
}

// m  n k <= 1024
extern "C" void solve(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) {
    dim3 threads = dim3(32, 32);
    dim3 blocks = dim3((N + BN - 1) / BN , (M + BM - 1) / BM , BATCH);

    matmul<<<blocks, threads>>>(A, B, C, BATCH, M, N, K);
    cudaDeviceSynchronize();
}
