// d is scaled from 128 to 1024. how do you handle it? solution handling d <= 1024; 
#include <cuda_runtime.h>
#include <float.h>

float static __device__ warp_reduce(float val) {
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, i);
    }
    return val;
}

void __global__ to_be_flash(const float* Q, const float* K, const float* V, float* output, int M, int N, int d, float alpha) {
    int m = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ float t[32];
    __shared__ float p, curr_mx, sum, correction, mx;
    if (tid == 0) {
        curr_mx = -FLT_MAX;
        sum = 0.0f;
        mx = -FLT_MAX;
    }
    float out = 0;
    for (int i = 0; i < N; i++) {
        float partial = tid < d ? Q[m * d + tid] * K[i * d + tid] : 0;
        // warp reduction
        partial = warp_reduce(partial);
        if (tid % 32 == 0) t[tid / 32] = partial;
        __syncthreads();
        if (tid / 32 == 0) {
            partial = t[tid % 32];
            partial = warp_reduce(partial);            
        }
        if (tid == 0) {
            p = partial / sqrt(d) + (m - i) * alpha;
            curr_mx = max(mx, p);
            correction = exp(mx - curr_mx);
            mx = curr_mx;
            sum = sum * correction + exp(p - mx);
        }
        __syncthreads();
        out = tid < d ? out * correction +  exp(p - mx) * V[i * d + tid] : out * correction;
    }
    if (tid < d)
    output[m*d + tid] = out / sum;
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d, float alpha) {
    int blocks = M;
    int threads = 1024;
    to_be_flash<<<blocks, threads>>>(Q,K,V,output, M, N,d,alpha);
    cudaDeviceSynchronize();
}
