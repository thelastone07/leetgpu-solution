#include <cuda_runtime.h>
#include <float.h>

__global__ void sum_and_max(const float* input, float* max, float* sum, int N) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    
    float local_max = -FLT_MAX, local_sum = 0;
    for (int i = 0; i < 32; i++) {
        if (i * blockDim.x + x < N) {
            float curr_max = fmax(input[x + i*blockDim.x],-FLT_MAX);
            float new_max = fmax(local_max, curr_max);
            local_sum = local_sum * exp(local_max - new_max) + exp(curr_max - new_max);
            local_max = new_max;
        }
    }
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    for (int i = 16; i >= 1; i = i / 2) {
        float other_max = __shfl_down_sync(0xffffffff, local_max, i);
        float other_sum = __shfl_down_sync(0xffffffff, local_sum, i);
           
        if (lane < i) {
            float new_max = fmax(other_max, local_max);
            local_sum = local_sum * exp(local_max - new_max) + other_sum * exp(other_max - new_max);
            local_max = new_max;
        }  
    }

    __shared__ float t_max[32], t_sum[32];
    if (lane == 0) {
        t_max[warp] = local_max;
        t_sum[warp] = local_sum;
    }
    __syncthreads();

    for (int i = 16; i >= 1; i /= 2) {
        if (warp < i && lane == 0) {
            float other_max = t_max[warp + i];
            float other_sum = t_sum[warp + i];
            float new_max = fmax(other_max, t_max[warp]);
            t_sum[warp] = (t_sum[warp]) * exp(t_max[warp] - new_max) + other_sum * exp(other_max - new_max);
            t_max[warp] = new_max;
        }
        __syncthreads();
    }
    if (warp == 0 && lane == 0) {
        max[blockIdx.x] = t_max[0];
        sum[blockIdx.x] = t_sum[0];
    }
}

__global__ void softmax(const float * input, float * output, float* max, float* sum, int N, int blocks) {
    int x = threadIdx.x;
    __shared__ float m, s;
    if (x == 0) {
        m = max[0]; s = sum[0];
        for (int i = 1;i < blocks; i++) {
            float new_max = fmax(m, max[i]);
            s =  s * exp(m - new_max) + sum[i] * exp(max[i] - new_max);
            m = new_max;
        }
    }
    __syncthreads();
    int i = 0;
    while (x + i * blockDim.x < N) {
        output[x+i*blockDim.x] = exp(input[x+i*blockDim.x] - m) / s;
        i++;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock*32 - 1) / (threadsPerBlock * 32);

    float *d_max, *d_sum;
    cudaMalloc(&d_max, sizeof(float) * blocksPerGrid);
    cudaMalloc(&d_sum, sizeof(float) * blocksPerGrid);
    sum_and_max<<<blocksPerGrid, threadsPerBlock>>>(input,d_max, d_sum, N);
    cudaDeviceSynchronize();

    softmax<<<1, 1024>>>(input, output, d_max, d_sum, N, blocksPerGrid);
    cudaDeviceSynchronize();

    cudaFree(d_max); cudaFree(d_sum);
}
