#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < rows && col < cols) {
        output[col * rows + row] = input[row  * cols + col];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}


int main() {
    int rows = 3, cols = 2;
    float h_input[] = {1.0f, 2.0f,
                       3.0f, 4.0f,
                       5.0f, 6.0f};
    float h_output[6] = {0};

    float *d_input, *d_output;
    cudaMalloc(&d_input, rows * cols * sizeof(float));
    cudaMalloc(&d_output, rows * cols * sizeof(float));

    cudaMemcpy(d_input, h_input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    solve(d_input, d_output, rows, cols);

    cudaMemcpy(h_output, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Raw output: ");
    for (int i = 0; i < 6; i++) printf("%.1f ", h_output[i]);
    printf("\n");


    printf("Output (%d x %d):\n", cols, rows);
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            printf("%.1f ", h_output[i * rows + j]);
        }
        printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}