#include <cuda_runtime.h>

__global__ void calc(const float* logits, const int* true_labels, float* loss, int N, int C) {
    int r = blockIdx.x;
    int idx = threadIdx.x;

    if (r >= N) return;

    float sum = 0;

    for (int i = idx; i < C; i += blockDim.x) {
        sum += exp(logits[r*C + i]);
    }

    for (int i = 16; i >= 1; i/=2) {
        sum += __shfl_down_sync(0xffffffff, sum, i);   
    }
    __shared__ float w_sum[8];
    if ( idx % 32 == 0) w_sum[idx / 32] = sum;

    __syncthreads();
    if (idx == 0) {
        sum = 0;
        for (int i = 0; i < 8; i++) sum += w_sum[i];
        loss[r] = log(sum) - logits[r*C + true_labels[r]];
    }
}

__global__ void reduce(float* unreduced_loss, float* loss, int N) {
    float sum = 0;
    for (int i = threadIdx.x; i < N; i+= blockDim.x) {
        sum += unreduced_loss[i];
    }

    for (int i = 16; i >= 1; i /= 2) sum += __shfl_down_sync(0xffffffff, sum, i);

    __shared__ float w_sum[32];
    int idx = threadIdx.x;
    if (idx % 32 == 0) w_sum[idx / 32] = sum;
    __syncthreads();
    if (idx / 32 == 0) {
        sum = w_sum[idx];
        for (int i = 16; i >= 1; i /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, i);
        }
    }

    if (idx == 0) *loss = sum / N;
}

// logits, true_labels, loss are device pointers
extern "C" void solve(const float* logits, const int* true_labels, float* loss, int N, int C) {
    int threadsPerBlock = 256;
    int blocksPerGrid = N;

    float* unreduced_loss;
    cudaMalloc(&unreduced_loss, sizeof(float)*N);
    calc<<<blocksPerGrid, threadsPerBlock>>>(logits, true_labels, unreduced_loss, N, C);
    cudaDeviceSynchronize();

    reduce<<<1, 1024>>>(unreduced_loss, loss, N);
    cudaDeviceSynchronize();

    cudaFree(unreduced_loss);
}
