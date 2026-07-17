#include<cuda_runtime.h>
#include<float.h>

float static __device__ warp_reduce(float val) {
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, i);
    }
    return val;
}

void __global__ almost_flash(const float* Q, const float* K, const float* V, float* output, int M, int N, int d, float alpha) {
    int m = blockIdx.x * blockDim.y + threadIdx.y ;
    int tid = threadIdx.x;

    int idx = blockDim.x * threadIdx.y + threadIdx.x;

    float qd[32];
    for (int i = 0; i < 32; i++) {
        qd[i] = (tid*32 + i < d && m < M) ? Q[m * d + tid* 32 + i] : 0.0f;
    }

    const int BC = 4;
    __shared__ float k[BC][1024], v[BC][1024];
    __shared__ float score[8][BC]; // per warp per key
    __shared__ float p[8][BC], mx[8], sum[8], correction[8];
    float out[32] = {0}, scale = 1 / sqrt(d);
    if (tid == 0) {
        mx[threadIdx.y] = -FLT_MAX, sum[threadIdx.y] = 0; 
    }
    __syncwarp();
    for (int n = 0; n < N; n += BC ) {
        // load the k and v
        for (int i = 0; i < BC; i++) {
            for (int c = idx; c < 1024; c += blockDim.x * blockDim.y) {
                k[i][c] = (n + i < N && c < d) ? K[(n+i)*d + c] : 0.0f;
                v[i][c] = (n + i < N && c < d) ? V[(n+i)*d + c] : 0.0f;
            }
        }

        __syncthreads();

        //compute the multiplication using loops 

        for (int i = 0; i < BC; i++) {
            float val = 0;
            for (int j = 0; j < 32; j++) {
                val += k[i][j + tid*32] * qd[j];
            }
            val = warp_reduce(val);
            if (tid == 0) {
                score[threadIdx.y][i] = (n + i < N) ? val * scale + (m - n - i) * alpha : -FLT_MAX;
            }
            
        }
        __syncwarp();
        if (tid == 0){ 
            float tile_max= -FLT_MAX;
            for (int i = 0; i < BC; i++) tile_max = fmaxf(tile_max, score[threadIdx.y][i]);

            float new_mx = fmaxf(mx[threadIdx.y], tile_max);
            float corr = expf(mx[threadIdx.y]-new_mx);
            correction[threadIdx.y] = corr;

            float tile_sum = 0;
            for (int i = 0; i < BC; i++) {
                float pi = expf(score[threadIdx.y][i] - new_mx);
                p[threadIdx.y][i] = pi;
                tile_sum += pi;
            }  
            
            sum[threadIdx.y] = sum[threadIdx.y] *corr + tile_sum;
            mx[threadIdx.y] = new_mx;
        }
        __syncwarp();

        float corr = correction[threadIdx.y];
        for (int j = 0; j < 32; j++) {
            float acc = out[j] * corr;
            for (int i = 0; i < BC; i++) {
                acc += p[threadIdx.y][i] * v[i][j + tid * 32];
            }
            out[j] = acc;
        }
        __syncthreads();
    }
    if (tid == 0) {
        sum[threadIdx.y] = 1 / sum[threadIdx.y];
    }
    __syncwarp();
    for (int j = 0; j < 32; j++) {
        if (tid * 32 + j < d && m < M) output[m * d + (tid * 32 + j)] = out[j] * sum[threadIdx.y];
    }
}


extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d, float alpha) {
    dim3 threads = dim3(32,8);
    int blocks = (M + 7) / 8;
    almost_flash<<<blocks, threads>>>(Q, K, V, output, M, N, d, alpha);
    cudaDeviceSynchronize();
}