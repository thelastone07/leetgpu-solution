#include <cuda_runtime.h>

__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K)  {
    const int TILE = 32;
    __shared__ float a[32][32], b[32][32];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0;

    for (int k = 0; k < (K + 31) / 32; k++) {
        int ak = TILE * k + threadIdx.x;
        int bk = TILE * k + threadIdx.y;

        a[threadIdx.y][threadIdx.x] = (ak < K && row < M) ? A[row * K + ak] : 0.0f;
        b[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? B[bk * N + col] : 0.0f;
        __syncthreads();

        for (int kk = 0; kk < 32; kk++) {
            sum += a[threadIdx.y][kk] * b[kk][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum; 
    }   
}

__global__ void gradient_atomic(const float* X, const float* p, const float* y, float* g,
                                 int n_samples, int n_features, int chunk_size) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_features) return;

    int row0 = blockIdx.y * chunk_size;              // sample-chunk owned by this y-slice
    int end = min(row0 + chunk_size, n_samples);

    float partial = 0;
    for (int i = row0; i < end; i++) {
        partial += X[i*n_features + j] * (p[i] - y[i]);
    }
    atomicAdd(&g[j], partial);
}

__global__ void gradient_finalize(float* g, const float* beta, int n_features, int n_samples, float lambda) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_features) return;
    g[j] = (g[j] + lambda * beta[j]) / n_samples;
}

__global__ void hessian_atomic(const float* X, const float* w, float* H,
                                int n_samples, int n_features, int chunk_size) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // feature row
    int k = blockIdx.x * blockDim.x + threadIdx.x;  // feature col
    if (j >= n_features || k >= n_features || j > k) return;  // upper triangle only

    int row0 = blockIdx.z * chunk_size;              // sample-chunk owned by this z-slice
    int end = min(row0 + chunk_size, n_samples);

    float partial = 0;
    for (int i = row0; i < end; i++) {
        partial += X[i*n_features + j] * w[i] * X[i*n_features + k];
    }

    atomicAdd(&H[j*n_features + k], partial);
    if (j != k) atomicAdd(&H[k*n_features + j], partial);  // symmetry, free
}

__global__ void hessian_finalize(float* H, int n_features, int n_samples, float reg) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_features || k >= n_features) return;

    float v = H[j*n_features + k] / n_samples;
    if (j == k) v += reg;
    H[j*n_features + k] = v;
}

template <int tile_size>
__global__ void mat_transpose_kernel(const float* A, float* At, int rows, int cols) {
    __shared__ float tile[tile_size][tile_size + 1];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        tile[threadIdx.y][threadIdx.x] = A[row * cols + col];
    }
    __syncthreads();

    row = blockIdx.x * blockDim.x + threadIdx.y;
    col = blockIdx.y * blockDim.y + threadIdx.x;
    if (row < cols && col < rows) {
        At[row * rows + col] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void sigmoid(float *z, float* p, float* w , int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        float pi = 1.0f / (1.0f + expf(-z[tid]));
        p[tid] = pi;
        w[tid] = pi * (1.0f - pi);
    }
}

__global__ void gauss_jordan(float* A, float* B, int N, float* beta) {
    int tid = threadIdx.x;
    __shared__ float pivot;
    for (int k = 0; k < N; k++) {
        if (tid == k) pivot = A[k*N + k];
        __syncthreads();
        if (tid < N)
        A[k*N + tid] = A[k*N + tid] / pivot;
        if (tid == 0) B[k] = B[k] / pivot;
        __syncthreads();
        
        if (tid != k && tid < N) {
            float factor = A[tid*N + k];
            for (int col = 0; col < N; col++) {
                A[tid*N + col] -= factor * A[k*N + col];
            }
            B[tid] -= factor * B[k];
        }
        __syncthreads();
    }
    if (tid < N) {
        beta[tid] = beta[tid] - B[tid] / A[tid*N + tid];
    }
}


// X, y, beta are device pointers
extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {
    float *z, *p, *w, *g, *H;
    cudaMalloc(&z, sizeof(float) * n_samples);
    cudaMalloc(&p, sizeof(float) * n_samples);
    cudaMalloc(&w, sizeof(float) * n_samples);
    cudaMalloc(&g, sizeof(float) * n_features);
    cudaMalloc(&H, sizeof(float) * n_features * n_features);

    cudaMemset(beta, 0, sizeof(float) * n_features);

    const float lambda = 1e-6f;
    const float reg     = 1e-6f;
    const int N_ITERS    = 30;
    const int CHUNK      = 1024;
    int n_chunks = (n_samples + CHUNK - 1) / CHUNK;

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaEvent_t sigmoid_done, hessian_done;
    cudaEventCreate(&sigmoid_done);
    cudaEventCreate(&hessian_done);

    const int threads1d = 256;
    dim3 block2d(16, 16);
    dim3 grid_feat2d((n_features + 15) / 16, (n_features + 15) / 16);

    for (int iter = 0; iter < N_ITERS; iter++) {
        // z = X*beta, then p, w  -- stream1
        dim3 block_z(32, 32);
        dim3 grid_z(1, (n_samples + 31) / 32);
        matmul<<<grid_z, block_z, 0, stream1>>>(X, beta, z, n_samples, 1, n_features);
        sigmoid<<<(n_samples + threads1d - 1) / threads1d, threads1d, 0, stream1>>>(z, p, w, n_samples);
        cudaEventRecord(sigmoid_done, stream1);

        // g pipeline -- stays on stream1
        cudaMemsetAsync(g, 0, sizeof(float) * n_features, stream1);
        dim3 grid_g((n_features + threads1d - 1) / threads1d, n_chunks);
        gradient_atomic<<<grid_g, threads1d, 0, stream1>>>(X, p, y, g, n_samples, n_features, CHUNK);
        gradient_finalize<<<(n_features + threads1d - 1) / threads1d, threads1d, 0, stream1>>>(g, beta, n_features, n_samples, lambda);

        // H pipeline -- stream2, waits only for p/w from sigmoid
        cudaStreamWaitEvent(stream2, sigmoid_done);
        cudaMemsetAsync(H, 0, sizeof(float) * n_features * n_features, stream2);
        dim3 grid_h(grid_feat2d.x, grid_feat2d.y, n_chunks);
        hessian_atomic<<<grid_h, block2d, 0, stream2>>>(X, w, H, n_samples, n_features, CHUNK);
        hessian_finalize<<<grid_feat2d, block2d, 0, stream2>>>(H, n_features, n_samples, reg);
        cudaEventRecord(hessian_done, stream2);

        // solve H*step = g and fold into beta -- needs both g (stream1) and H (stream2)
        cudaStreamWaitEvent(stream1, hessian_done);
        gauss_jordan<<<1, 1024, 0, stream1>>>(H, g, n_features, beta);
    }

    cudaStreamSynchronize(stream1);

    cudaEventDestroy(sigmoid_done);
    cudaEventDestroy(hessian_done);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(z); cudaFree(p); cudaFree(w);
    cudaFree(g); cudaFree(H);
}
