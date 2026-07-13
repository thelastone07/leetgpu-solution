#include <cuda_runtime.h>

__global__ void block_level(float* input, float* output, float* aux, int N) {
    int i = threadIdx.x  + blockDim.x * blockIdx.x ;
    int tid = threadIdx.x;

    float val = i >= N ? 0 : input[i];
    int lane = tid % 32;
    for (int j = 1; j < 32; j = j << 1) {
        float val_ = __shfl_up_sync(0xffffffff, val, j);
        if (lane >= j) val += val_;
    }

    __shared__ float t[32];
    if (lane == 31)
    t[tid / 32] = val;

    __syncthreads();

    if (tid / 32 == 0) {
        float other_val = t[lane];
        for (int j = 1; j < 32; j = j << 1) {
            float val_ = __shfl_up_sync(0xffffffff, other_val, j);
            if (lane >= j) other_val += val_;
        }
        t[lane] = other_val;
    }
    __syncthreads();

    float prefix = (tid / 32 == 0) ? 0.0f : t[tid / 32 - 1];
    if (i < N)
    output[i] = val + prefix;

    if (tid == 1023)
    aux[blockIdx.x] = val + prefix;
}

__global__ void block_level_without_aux(float* input, float* output, int N) {
    int i = threadIdx.x  + blockDim.x * blockIdx.x ;
    int tid = threadIdx.x;

    float val = i >= N ? 0 : input[i];
    int lane = tid % 32;
    for (int j = 1; j < 32; j = j << 1) {
        float val_ = __shfl_up_sync(0xffffffff, val, j);
        if (lane >= j) val += val_;
    }

    __shared__ float t[32];
    if (lane == 31)
    t[tid / 32] = val;

    __syncthreads();

    if (tid / 32 == 0) {
        float other_val = t[lane];
        for (int j = 1; j < 32; j = j << 1) {
            float val_ = __shfl_up_sync(0xffffffff, other_val, j);
            if (lane >= j) other_val += val_;
        }
        t[lane] = other_val;
    }
    __syncthreads();
    float prefix = (tid / 32 == 0) ? 0.0f : t[tid / 32 - 1];
    if (i < N)
        output[i] = val + prefix;

}

__global__ void add(float* output, float* sum, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int bid = blockIdx.x;

    if (bid >= 1 && i < N) {
        output[i] += sum[bid-1];
    }
}


// input, output are device pointers
void solver(float* input, float* output, int N) {
    int threads = 1024;
    int blocks = (N + 1023) / 1024;
    float *aux;
    cudaMalloc(&aux, blocks*sizeof(float));
    if (blocks == 1) {
        block_level_without_aux<<<blocks, threads>>>(input, output, N);
        cudaDeviceSynchronize();
    } else {
        block_level<<<blocks, threads>>>(input, output, aux, N);
        if (blocks <= 1024) {
            float *block_sum;
            cudaMalloc(&block_sum, blocks * sizeof(float));
            block_level_without_aux<<<1, threads>>>(aux, block_sum, blocks);
            cudaDeviceSynchronize();
            add<<< blocks, threads>>>(output, block_sum, N);
            cudaDeviceSynchronize();
            cudaFree(block_sum);
        }
        else {
            float *out;
            cudaMalloc(&out, sizeof(float)*blocks);
            solver(aux, out, blocks);
            add<<< blocks, threads>>>(output, out, N);
            cudaDeviceSynchronize();
            cudaFree(out);
        }
    }
    cudaFree(aux);
}

extern "C" void solve(const float* input, float* output, int N) {

    float *inp;
    cudaMalloc(&inp, sizeof(float)*N);
    cudaMemcpy(inp, input, N * sizeof(float), cudaMemcpyDeviceToDevice);
    solver(inp, output, N);

    cudaFree(inp);
}