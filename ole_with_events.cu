#include <cuda_runtime.h>

void __global__ tranpose(const float* X, float* output, int R, int C) {
    __shared__ float tile[32][33];
    int c = threadIdx.x + blockIdx.x * blockDim.x;
    int r = threadIdx.y + blockIdx.y * blockIdx.y;
    
    if (r < R && c < C) tile[threadIdx.y][threadIdx.x] = X[r * C + c];
    __syncthreads();
    r = threadIdx.y + blockIdx.x * blockDim.x;
    c = threadIdx.x + blockIdx.y * blockDim.y;
    if (r < C && c < R) output[r * R + c] = tile[threadIdx.x][threadIdx.y];
}

void __global__ matmul(float* A, const float* B, float* output, int M, int N, int K) {
    __shared__ float aa[32][32], bb[32][32]; 
    float sum = 0;
    for (int bk = 0; bk < (K + 31) / 32; bk++) {
        int r = blockIdx.y * blockDim.y + threadIdx.y;
        int c = bk * 32 +threadIdx.x; 
        aa[threadIdx.y][threadIdx.x] = (r < M && c < K) ? A[r * K + c] : 0.0f;

        r = bk * 32 + threadIdx.y;
        c = blockIdx.x * blockDim.x + threadIdx.x;
        bb[threadIdx.y][threadIdx.x] = (r < K && c < N) ? B[r * N + c] : 0.0f;

        __syncthreads();
        
        for (int k = 0; k < 32; k++) {
            sum += aa[threadIdx.y][k] * bb[k][threadIdx.x];
        }
        __syncthreads();
    }
    int c = threadIdx.x + blockIdx.x * blockDim.x;
    int r = threadIdx.y + blockIdx.y * blockDim.y;
    if (c < N && r < M) {
        output[r * N + c] = sum;
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
        beta[tid] = B[tid] / A[tid*N + tid];
    }
}

// X, y, beta are device pointers
extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {
    float *XT, *A, *B;
    cudaMalloc(&XT, sizeof(float) * n_features * n_samples);
    cudaMalloc(&A, sizeof(float) * n_features * n_features);
    cudaMalloc(&B, sizeof(float) * n_features);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaEvent_t trans_done, b_done;
    cudaEventCreate(&trans_done);
    cudaEventCreate(&b_done);

    dim3 threads(32, 32);

    // XT = X^T   (n_features x n_samples)
    dim3 trans_grid((n_features + 31) / 32, (n_samples + 31) / 32);
    tranpose<<<trans_grid, threads, 0, stream1>>>(X, XT, n_samples, n_features);
    cudaEventRecord(trans_done, stream1);

    // stream2: B = X^T * y   (n_features x 1) -- waits on transpose
    cudaStreamWaitEvent(stream2, trans_done);
    dim3 b_grid((1 + 31) / 32, (n_features + 31) / 32);
    matmul<<<b_grid, threads, 0, stream2>>>(XT, y, B, n_features, 1, n_samples);
    cudaEventRecord(b_done, stream2);

    // stream1: A = X^T * X   (n_features x n_features) -- already ordered after transpose
    dim3 a_grid((n_features + 31) / 32, (n_features + 31) / 32);
    matmul<<<a_grid, threads, 0, stream1>>>(XT, X, A, n_features, n_features, n_samples);

    // gauss_jordan needs both A (stream1, already ordered) and B (stream2 -- explicit wait)
    cudaStreamWaitEvent(stream1, b_done);
    gauss_jordan<<<1, 1024, 0, stream1>>>(A, B, n_features, beta);

    cudaStreamSynchronize(stream1);

    cudaEventDestroy(trans_done);
    cudaEventDestroy(b_done);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(XT);
    cudaFree(A);
    cudaFree(B);
}
