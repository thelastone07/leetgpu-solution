#include <cuda_runtime.h>
#include<float.h>

template <int CHUNK_SIZE>
__global__ void nearest_neighbour(const float* points, int* indices, int N) {
    __shared__ float3 local_points[CHUNK_SIZE];

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    float sum = FLT_MAX, best = FLT_MAX;
    int idx = -1;
    float3 point;
    if (i < N) point = reinterpret_cast<const float3*>(points)[i];
    for (int j = 0; j < N; j += CHUNK_SIZE) {
        if (tid + j < N)
        local_points[tid] = reinterpret_cast<const float3*>(points)[tid+j];
        __syncthreads();
        if (i < N) {
            for (int k = 0; k < CHUNK_SIZE; k++) {
                if (i == j+k) continue;
                if (j + k < N) {
                    sum = (point.x - local_points[k].x)*(point.x - local_points[k].x) + (point.y - local_points[k].y)*(point.y - local_points[k].y) + (point.z - local_points[k].z)*(point.z - local_points[k].z);
                    if (sum < best) {
                        best = sum;
                        idx = j + k;
                    }
                    sum = FLT_MAX;
                }
            }
        }
        __syncthreads();
    }

    if (i < N) indices[i] = idx;
}

// points and indices are device pointers
extern "C" void solve(const float* points, int* indices, int N) {
    int threads = 1024;
    constexpr int CHUNK_SIZE = 1024;
    int block = ( N + 1023) / 1024;
    nearest_neighbour<CHUNK_SIZE><<<block, threads>>>(points, indices, N);
    cudaDeviceSynchronize();
}
