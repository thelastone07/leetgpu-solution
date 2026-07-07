#include <cuda_runtime.h>

__global__ void conv(const float* input, const float* kernel, float* output, int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    __shared__ float k[31][31];
    if (threadIdx.x < kernel_cols && threadIdx.y < kernel_rows)
    k[threadIdx.y][threadIdx.x] = kernel[threadIdx.y*kernel_cols + threadIdx.x];

    int inp_rows = blockDim.y + kernel_rows - 1; // 32 + 31 - 1 = 62
    int inp_cols = blockDim.x + kernel_cols - 1;
    __shared__ float inp[62][62];

    int r = blockDim.y * blockIdx.y + threadIdx.y;
    int c = blockDim.x * blockIdx.x + threadIdx.x;
    
    
    for (int i = threadIdx.y; i < inp_rows; i+=blockDim.y) {
        for (int j = threadIdx.x; j < inp_cols; j+=blockDim.x) {
            inp[i][j] = (r - threadIdx.y + i < input_rows && c - threadIdx.x + j < input_cols) ? input[(r - threadIdx.y + i) * input_cols + (c - threadIdx.x + j)] : 0.0f;
        }
    }

    __syncthreads();
    int out_rows = input_rows - kernel_rows + 1;
    int out_cols = input_cols - kernel_cols + 1;
    if (r < out_rows && c < out_cols) {
        float out = 0;
        for (int i = 0; i < kernel_rows; i++) {
            for (int j = 0; j < kernel_cols; j++) {
                out += inp[i + threadIdx.y][j + threadIdx.x] * k[i][j];
            }
        }
        output[r * out_cols + c] = out;
    }

}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    int out_rows = input_rows - kernel_rows + 1;
    int out_cols = input_cols - kernel_cols + 1;
    
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid(
        (out_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (out_rows + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    conv<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
}
