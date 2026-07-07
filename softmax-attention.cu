#include<cuda_runtime.h>
#include<float.h>

__global__ void flash(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    int i = blockIdx.x;
    int j = threadIdx.x;

    __shared__ float q[128];

    if (j < d) {
        q[j] = Q[i*d + j];
    }
    __syncthreads();

    float local_max = -FLT_MAX, local_sum=0.0f, local_output[128] = {0};
    float scale = 1 / sqrt(d);
    
    int jj = j;
    while (jj < N) {
        float score = 0;
        for (int k = 0; k < d; k++) {
            score += q[k] * K[jj * d + k];
        }
        score *= scale;
        float new_max = fmax(score, local_max);
        float correction = exp(local_max - new_max);
        local_sum = local_sum * correction + exp(score - new_max);
        for (int k = 0; k < d; k++) {
            local_output[k] = local_output[k] * correction + exp(score - new_max) * V[jj * d + k];
        }
        local_max = new_max;
        jj += blockDim.x;
    }

    int warp = j / 32;
    int lane = j % 32;

    for (int z = 16; z >= 1; z /= 2) {
        float other_max = __shfl_down_sync(0xffffffff, local_max, z);
        float other_sum = __shfl_down_sync(0xffffffff, local_sum, z);
        float new_max;
        if (lane < z) {
            new_max = fmax(other_max, local_max);
            local_sum = local_sum * exp(local_max - new_max) + other_sum * exp(other_max - new_max);
        }
        for (int k = 0; k < d; k++) {
            float other_output = __shfl_down_sync(0xffffffff, local_output[k], z);
            if (lane < z) {
                local_output[k] = local_output[k] * exp(local_max - new_max) + other_output * exp(other_max - new_max);
            }
        }
        if (lane < z)
        local_max = new_max;
    }

    __shared__ float t_max[32], t_sum[32], t_output[32][128];
    if (lane == 0) {
        t_max[warp] = local_max;
        t_sum[warp] = local_sum;
        for (int z = 0; z < d; z++) {
            t_output[warp][z] = local_output[z];
        }
    }

    __syncthreads();
    for (int z = 16; z >= 1; z /= 2) {
        if (j < z) {
            float new_max = fmax(t_max[j], t_max[j+z]);
            t_sum[j] = t_sum[j] * exp(t_max[j] - new_max) + t_sum[j + z] * exp(t_max[j+z] - new_max);
            for (int k = 0; k < d; k++) {
                t_output[j][k] = t_output[j][k] * exp(t_max[j] - new_max) + t_output[j+z][k] * exp(t_max[j+z] - new_max); 
            }
            t_max[j] = new_max;
        }
        __syncthreads();
    }
    if (j == 0) {
        float scale = 1 / t_sum[0];
        for (int k = 0; k < d; k++) {
            output[i * d + k] = t_output[0][k] * scale ;
        }
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    int threadsPerBlock = 1024;
    int blocksPerGrid = M;

    flash<<<blocksPerGrid, threadsPerBlock>>>(Q, K, V, output, M, N, d);
    cudaDeviceSynchronize(); 
}