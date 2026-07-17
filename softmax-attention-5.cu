// considering d <= 128 - realistic and then i will jump into wmma

#include<cuda_runtime.h>
#include<float.h>

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

void __global__ almost_flash(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    int warp = threadIdx.y;
    int tid = threadIdx.x;
    int idx = blockDim.x * threadIdx.y + threadIdx.x;

    int row_base = blockIdx.x * (WARPS * ROWS_PER_WARP) + warp * ROWS_PER_WARP;

    float qd[ROWS_PER_WARP][4];
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        int global_row = row_base + r;
        for (int i = 0; i < 4; i++) {
            qd[r][i] = (tid * 4 + i < d && global_row < M) ? Q[global_row * d + tid * 4 + i] : 0.0f;
        }
    }

    __shared__ float k[BC][128], v[BC][128];
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
                k[i][c] = (n + i < N && c < d) ? K[(n+i)*d + c] : 0.0f;
                v[i][c] = (n + i < N && c < d) ? V[(n+i)*d + c] : 0.0f;
            }
        }

        __syncthreads();

        //compute the multiplication using loops 

        for (int r = 0; r < ROWS_PER_WARP; r++) {
            for (int i = 0; i < BC; i++) {
                float val = 0;
                for (int j = 0; j < 4; j++) {
                    val += k[i][j + tid * 4] * qd[r][j];
                }
                val = warp_reduce(val);
                if (tid == 0) {
                    score[warp][r][i] = (n + i < N) ? val * scale : -FLT_MAX;
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
    almost_flash<<<blocks, threads>>>(Q, K, V, output, M, N, d);
    cudaDeviceSynchronize();
}