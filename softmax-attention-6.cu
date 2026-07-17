// considering d <= 128 - realistic and then i will jump into wmma

#include<cuda_runtime.h>
#include<float.h>
#include<mma.h>
using namespace nvcuda;

float static __device__ warp_reduce(float val) {
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, i);
    }
    return val;
}

const int ROWS_PER_WARP = 16;
const int WARPS = 4;
const int BC = 16;

void __global__ flash(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    int warp = threadIdx.y;
    int tid = threadIdx.x;
    int idx = blockDim.x * threadIdx.y + threadIdx.x;

    int row_base = blockIdx.x * (WARPS * ROWS_PER_WARP) + warp * ROWS_PER_WARP;

    __shared__ half q_half[WARPS][16][128], k_half[BC][128];

    for (int r = 0; r < ROWS_PER_WARP; r++) {
        int global_row = row_base + r;
        for (int c = tid; c < 128; c += 32) {
            q_half[warp][r][c] = (c < d && global_row < M) ? __float2half(Q[global_row * d + c]) : __float2half(0.0f);
        }
    }
    __syncwarp();

    __shared__ float v[BC][128];
    __shared__ float score[WARPS][ROWS_PER_WARP][BC]; // per warp per key
    __shared__ float p[WARPS][ROWS_PER_WARP][BC], mx[WARPS][ROWS_PER_WARP], sum[WARPS][ROWS_PER_WARP], correction[WARPS][ROWS_PER_WARP];
    
    float out[ROWS_PER_WARP][4] = {0}, scale = 1.0 / sqrt((float)d);

    if (tid == 0) {
        for (int r = 0; r < ROWS_PER_WARP; r++) {
            mx[warp][r] = -FLT_MAX, sum[warp][r] = 0.0f;
        }
    }
    __syncwarp();

    for (int n = 0; n < N; n += BC ) {
        // load the k and v
        for (int i = 0; i < BC; i++) {
            for (int c = idx; c < 128; c += blockDim.x * blockDim.y) {
                float kv = (n + i < N && c < d) ? K[(n+i)*d + c] : 0.0f;
                v[i][c] = (n + i < N && c < d) ? V[(n+i)*d + c] : 0.0f;

                k_half[i][c] = __float2half(kv);
            }
        }

        __syncthreads();

        //compute the multiplication using loops 

        wmma::fragment<wmma::matrix_a,16,16,16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b,16,16,16, half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16,16,16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        for (int dc = 0; dc < d; dc += 16) {
            wmma::load_matrix_sync(a_frag, &q_half[warp][0][dc], 128);
            wmma::load_matrix_sync(b_frag, &k_half[0][dc], 128);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        wmma::store_matrix_sync(&score[warp][0][0], c_frag, BC, wmma::mem_row_major);
        __syncwarp();

        if (tid == 0) {
            for (int r = 0; r < ROWS_PER_WARP; r++) {
                for (int i = 0; i < BC; i++) {
                    score[warp][r][i] = (n + i < N) ? score[warp][r][i] * scale : -FLT_MAX;
                }
            }
        }
        __syncwarp();

        if (tid == 0){ 
            for (int r = 0; r < ROWS_PER_WARP; r++) {
                float tile_max = -FLT_MAX;
                for (int i = 0; i < BC; i++) tile_max = max(tile_max, score[warp][r][i]);

                float new_mx = max(mx[warp][r], tile_max);
                float corr = exp(mx[warp][r] - new_mx);
                correction[warp][r] = corr;

                float tile_sum = 0;
                for (int i = 0; i < BC; i++) {
                    float pi = exp(score[warp][r][i] - new_mx);
                    p[warp][r][i] = pi;
                    tile_sum += pi;
                }
                sum[warp][r] = sum[warp][r] * corr + tile_sum;
                mx[warp][r] = new_mx;
            }
        }
        __syncwarp();

        for (int r = 0; r < ROWS_PER_WARP; r++) {
            float corr = correction[warp][r];
            for (int j = 0; j < 4; j++) {
                float acc = out[r][j] * corr;
                for (int i = 0; i < BC; i++) {
                    acc += p[warp][r][i] * v[i][j + tid * 4];
                }
                out[r][j] = acc;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        for (int r = 0; r < ROWS_PER_WARP; r++){
            sum[warp][r] = 1.0/ sum[warp][r];
        }
    }
    __syncwarp();
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        int global_row = row_base + r;
        for (int j = 0; j < 4; j++) {
            if (tid * 4 + j < d && global_row < M) {
                output[global_row * d + tid * 4 + j] = out[r][j] * sum[warp][r];
            }
        }
    }
}


extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    dim3 threads = dim3(32,WARPS);
    int blocks = (M + WARPS * ROWS_PER_WARP - 1) / (WARPS * ROWS_PER_WARP);
    flash<<<blocks, threads>>>(Q, K, V, output, M, N, d);
    cudaDeviceSynchronize();
}