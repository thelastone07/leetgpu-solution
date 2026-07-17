#include <cuda_runtime.h>
#include <float.h>

void __global__ d2_max_pool(const float* input, float* output, int N, int C, int H, int W, int H0, int W0, int kernel_size, int stride, int padding) {
    int wo = threadIdx.x + blockIdx.x * blockDim.x;
    int ho = threadIdx.y + blockIdx.y * blockDim.y;

    if (ho >= H0 || wo >= W0) return;
    int hi = ho*stride - padding;
    int wi = wo*stride - padding;

    int nc = blockIdx.z;
    int n = nc / C;
    int c = nc % C;

    float lmax = -FLT_MAX;
    const float *inp = input + (n* C + c) * H * W ;
    for (int i = 0; i < kernel_size; i++) {
        int hh = hi + i;
        if (hh >= H || hh < 0) continue;
        for(int j = 0; j < kernel_size; j++) {
            int ww = wi + j;
            if (ww >= W || ww < 0) continue;
            lmax = fmaxf(lmax, inp[hh*W + ww]);
        }
    }
    output[(n*C + c)*H0*W0 + ho*W0 + wo] = lmax;

}


// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N, int C, int H, int W, int kernel_size, int stride, int padding) {
   int ho = (H + 2*padding - kernel_size) / stride + 1;
   int wo = (W + 2*padding - kernel_size) / stride + 1;

   dim3 threads = dim3(16,16);
   dim3 blocks = dim3((wo + threads.x - 1) / threads.x, (ho + threads.y - 1) / threads.y, N * C);
    d2_max_pool<<<blocks, threads>>>(input, output, N, C, H, W, ho, wo, kernel_size, stride, padding);
    cudaDeviceSynchronize();
}
