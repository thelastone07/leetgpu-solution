#include <cuda_runtime.h>

__global__ void conv(const float* input, const float* kernel, float* output, int input_depth, int input_rows, int input_cols, int kernel_depth, int kernel_rows, int kernel_cols) {
    __shared__ float k[5][5][5];

    if (threadIdx.y < kernel_rows && threadIdx.x < kernel_cols && threadIdx.z < kernel_depth)
    k[threadIdx.z][threadIdx.y][threadIdx.x] = kernel[threadIdx.z * kernel_cols * kernel_rows + threadIdx.y * kernel_cols + threadIdx.x];

    int inp_rows = blockDim.y + kernel_rows - 1;
    int inp_cols = blockDim.x + kernel_cols - 1;
    int inp_depth = blockDim.z + kernel_depth - 1;

    int st_d = blockDim.z * blockIdx.z;
    int st_r = blockDim.y * blockIdx.y;
    int st_c = blockDim.x * blockIdx.x;

    __shared__ float inp[12][12][12];
    for (int d = threadIdx.z; d < inp_depth; d += blockDim.z) {
        for (int i = threadIdx.y; i < inp_rows; i += blockDim.y) {
            for (int j = threadIdx.x; j < inp_cols; j += blockDim.x) {
                inp[d][i][j] = (st_d + d < input_depth && st_c + j < input_cols && st_r + i < input_rows) ? input[(st_d + d ) * input_cols * input_rows + (st_r + i) * input_cols + (st_c + j)] : 0.0f;
            }
        }
    }

    __syncthreads();
    int d = st_d + threadIdx.z;
    int r = st_r + threadIdx.y;
    int c = st_c + threadIdx.x;

    int orr = input_rows - kernel_rows + 1;
    int oc = input_cols - kernel_cols + 1;
    int od = input_depth - kernel_depth + 1;

    if (d < od && r < orr && c < oc) {
        float out = 0;
        for (int dd = 0; dd < kernel_depth; dd++) {
            for (int i = 0; i < kernel_rows; i++) {
                for (int j = 0; j < kernel_cols; j++) {
                    out += k[dd][i][j] * inp[dd + threadIdx.z][i + threadIdx.y][j + threadIdx.x];
                }
            }
        }
        output[d * orr * oc + r * oc + c] = out;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_depth, int input_rows, int input_cols, int kernel_depth, int kernel_rows, int kernel_cols) {
    int out_rows = input_rows - kernel_rows + 1;
    int out_cols = input_cols - kernel_cols + 1;
    int out_depth = input_depth - kernel_depth + 1;
    
    dim3 threadsPerBlock(8,8,8);
    dim3 blocksPerGrid(
        (out_cols + 7) / 8,
        (out_rows + 7) / 8,
        (out_depth + 7) / 8
    );

    conv<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_depth, input_rows, input_cols, kernel_depth, kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
}
