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

__global__ void update(float* beta, float* g, float lr, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) beta[tid] += lr * g[tid];
}

__global__ void gradient_kernel(const float* X, float* p, const float* y, float* beta, float* g, int n_features, int n_samples, float lambda) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j < n_features) {
        float acc = 0;
        for (int i = 0; i < n_samples; i++) {
            acc += X[i*n_features + j] * (p[i] - y[i]) ;
        }
        g[j] = (acc + lambda * beta[j]) / n_samples;
    }
}

__global__ void hessian_kernel(const float* X, const float* w, float* H, int n_samples, int n_features, float reg) {
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (col >= n_features || row >= n_features) return;

    float acc = 0;
    for (int i = 0; i < n_samples; i++) {
        acc += X[i*n_features + row] * w[i] * X[i*n_features + col];
    }
    acc /= n_samples;
    if (row == col) acc += reg;
    H[row * n_features + col] = acc ;
}

void solve_linear_system(float* H, float* g, float* step, int n) {
    float* aug = new float[n * (n + 1)];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) aug[i*(n+1)+j] = H[i*n+j];
        aug[i*(n+1)+n] = g[i];
    }
    for (int k = 0; k < n; k++) {
        int pivot = k;
        for (int i = k+1; i < n; i++)
            if (fabsf(aug[i*(n+1)+k]) > fabsf(aug[pivot*(n+1)+k])) pivot = i;
        for (int j = 0; j <= n; j++) {
            float tmp = aug[k*(n+1)+j];
            aug[k*(n+1)+j] = aug[pivot*(n+1)+j];
            aug[pivot*(n+1)+j] = tmp;
        }
        for (int i = 0; i < n; i++) {
            if (i == k) continue;
            float f = aug[i*(n+1)+k] / aug[k*(n+1)+k];
            for (int j = k; j <= n; j++) aug[i*(n+1)+j] -= f * aug[k*(n+1)+j];
        }
    }
    for (int i = 0; i < n; i++) step[i] = aug[i*(n+1)+n] / aug[i*(n+1)+i];
    delete[] aug;
}

extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {
    float *z, *p, *w, *g, *H_dev, *step_dev;
    cudaMalloc(&z,       sizeof(float) * n_samples);
    cudaMalloc(&p,       sizeof(float) * n_samples);
    cudaMalloc(&w,       sizeof(float) * n_samples);
    cudaMalloc(&g,       sizeof(float) * n_features);
    cudaMalloc(&H_dev,   sizeof(float) * n_features * n_features);
    cudaMalloc(&step_dev, sizeof(float) * n_features);

    float* H_host    = new float[n_features * n_features];
    float* g_host    = new float[n_features];
    float* step_host = new float[n_features];

    cudaMemset(beta, 0, sizeof(float) * n_features);

    const float lambda = 1e-6f;
    const float reg    = 1e-6f;
    const int N_ITERS  = 30;
    int threads = 256;

    for (int iter = 0; iter < N_ITERS; iter++) {
        dim3 grid_s(1, (n_samples + 31) / 32);
        dim3 block(32, 32);
        matmul<<<grid_s, block>>>(X, beta, z, n_samples, 1, n_features);
        cudaDeviceSynchronize();

        sigmoid<<<(n_samples + threads - 1) / threads, threads>>>(z, p, w, n_samples);
        cudaDeviceSynchronize();

        gradient_kernel<<<(n_features + threads - 1) / threads, threads>>>(X, p, y, beta, g, n_features, n_samples, lambda);
        cudaDeviceSynchronize();

        dim3 block2(16, 16);
        dim3 grid_h((n_features + 15) / 16, (n_features + 15) / 16);
        hessian_kernel<<<grid_h, block2>>>(X, w, H_dev, n_samples, n_features, reg);
        cudaDeviceSynchronize();

        cudaMemcpy(H_host, H_dev, sizeof(float) * n_features * n_features, cudaMemcpyDeviceToHost);
        cudaMemcpy(g_host, g,     sizeof(float) * n_features,               cudaMemcpyDeviceToHost);

        solve_linear_system(H_host, g_host, step_host, n_features);

        cudaMemcpy(step_dev, step_host, sizeof(float) * n_features, cudaMemcpyHostToDevice);

        update<<<(n_features + threads - 1) / threads, threads>>>(beta, step_dev, -1.0f, n_features);
        cudaDeviceSynchronize();
    }

    delete[] H_host; delete[] g_host; delete[] step_host;
    cudaFree(z); cudaFree(p); cudaFree(w);
    cudaFree(g); cudaFree(H_dev); cudaFree(step_dev);
}