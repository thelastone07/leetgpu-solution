#include <cuda_runtime.h>

__global__ void gauss(const float* input, const float* kernel, float* output, int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    __shared__ float k[21][21], img[52][52];

    if (threadIdx.x < kernel_cols && threadIdx.y < kernel_rows) {
        k[threadIdx.y][threadIdx.x] = kernel[threadIdx.y * kernel_cols + threadIdx.x ];
    }

    __syncthreads();

    int st_r = blockDim.y * blockIdx.y - 10;
    int st_c = blockDim.x * blockIdx.x - 10;

    for (int i = threadIdx.y; i < 52; i += blockDim.y) {
        for (int j = threadIdx.x ; j < 52 ; j += blockDim.x) {
            int gi = st_r + i, gj = st_c + j;
            if (gi >= 0 && gj >= 0 && gi < input_rows && gj < input_cols)
            img[gi - st_r][gj - st_c] = input[gi * input_cols + gj];
            else img[gi-st_r][gj - st_c] = 0.0;
        }
    }

    __syncthreads();

    int img_r = 10 + threadIdx.y;
    int img_c = 10 + threadIdx.x;
    float sum = 0;
    for (int i = -kernel_rows / 2; i <= kernel_rows/2; i++) {
        for (int j = -kernel_cols / 2; j <= kernel_cols/2; j++) {
                sum += k[i + kernel_rows / 2][j + kernel_cols / 2] * img[img_r + i][img_c + j];
        }
    }
    int out_r = threadIdx.y + blockDim.y * blockIdx.y;
    int out_c = threadIdx.x + blockDim.x * blockIdx.x;
    if (out_r< input_rows && out_c < input_cols)
    output[ (out_r)* input_cols + out_c] = sum;
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    dim3 threads = dim3(32, 32);
    dim3 blocks = dim3((input_cols + 31) / 32, (input_rows + 31) / 32);

    gauss<<<blocks ,threads>>>(input, kernel, output, input_rows,  input_cols, kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
}
