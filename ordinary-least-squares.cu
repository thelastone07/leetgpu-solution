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

__global__ void transpose(const float* X, float* XT, int N, int M) {
    int c = threadIdx.x + blockDim.x * blockIdx.x;
    int r = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (r < N && c < M) {
        XT[N*c + r] = X[M*r + c];
    }
}

// X, y, beta are device pointers
extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {
    float *A, *B, *XT;
    
    cudaMalloc(&A, sizeof(float)*n_features * n_features);
    cudaMalloc(&B, sizeof(float) * n_features);
    cudaMalloc(&XT, sizeof(float)* n_features * n_samples);

    dim3 threadsPerBlock(32,32);
    dim3 blocksPerGrid((n_features + 31)/ 32, (n_samples+31)/32); // n x m
    transpose<<<blocksPerGrid, threadsPerBlock>>>(X, XT, n_samples, n_features);
    cudaDeviceSynchronize();

    blocksPerGrid = dim3((n_features + 31) / 32, (n_features + 31) / 32);
    matmul<<<blocksPerGrid, threadsPerBlock>>>(XT, X, A, n_features, n_features, n_samples);
    cudaDeviceSynchronize();

    matmul<<<blocksPerGrid, threadsPerBlock>>>(XT, y, B, n_features, 1, n_samples);
    cudaDeviceSynchronize();

    gauss_jordan<<<1, 1024>>>(A, B, n_features, beta);
    cudaDeviceSynchronize();

    cudaFree(A);
    cudaFree(B);
    cudaFree(XT);
}



// diff solution by some user 

#include <cuda_runtime.h>


static __device__ __host__ __forceinline__ int div_up(int a, int b) {
    return (a + b - 1) / b;
}

static __device__ float warp_all_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}


static __device__ float block_all_reduce_sum(float val) {
    __shared__ float ret;
    __shared__ float shared[32]; // Assuming max 1024 threads per block
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    val = warp_all_reduce_sum(val);
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[threadIdx.x] : 0.0f;
    val = warp_all_reduce_sum(val);

    if (threadIdx.x == 0) {
        ret = val;
    }
    __syncthreads();
    
    return ret;
}


// X^T
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

// X^T * y
// (X^T * X)^-1 * (X^T * y)
__global__ void mv_kernel(const float* xt, const float* y, float* Xty, int n) {
    int tid = blockIdx.x * n + threadIdx.x;
    int row = blockIdx.x;
    float val = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        val += xt[row * n + i] * y[i];
    }
    __syncthreads();

    val = block_all_reduce_sum(val);
    if (threadIdx.x == 0) {
        Xty[row] = val;
    }
}


// init d_xtx_inv to identity matrix
__global__ void init_identity_matrix_kernel(float* A, int n) {
    for (int i = threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        A[i * n + i] = 1.0f;
    }
}


// Gaussian elimination
__global__ void gaussian_elimination_kernel(float* A, float* AI, int n) { 
    // 循环每一列
    for (int col = 0; col < n; col++) {
        float pivot = A[col * n + col];
        // 第 col 行归一化
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            A[col * n + i] /= pivot;
            AI[col * n + i] /= pivot;
        }
        __syncthreads();

        // 每一行消元
        for (int row = 0; row < n; row++) {
            float factor = A[row * n + col];
            for (int i = threadIdx.x; i < n; i += blockDim.x) {
                if (row != col) {
                    A[row * n + i] -= factor * A[col * n + i];
                    AI[row * n + i] -= factor * AI[col * n + i];
                }
            }
        }
        __syncthreads();
    }
}


// X^T * X 
template <int tile_size>
__global__ void matrix_muliply_kernel(const float* A, const float* B, float* C, int m, int n, int k) {
    __shared__ float s_A[tile_size * tile_size];
    __shared__ float s_B[tile_size * tile_size];

    int tx = threadIdx.x % tile_size;
    int ty = threadIdx.x / tile_size;
    int row = blockIdx.y * tile_size + ty;
    int col = blockIdx.x * tile_size + tx;
    float acc = 0.0f;

    for (int i = 0; i < k; i += tile_size) {
        // load A and B from global memory to shared memory
        for (int j = threadIdx.x; j < tile_size * tile_size; j += blockDim.x) {
            int row_a = blockIdx.y * tile_size + j / tile_size;
            int col_a = i + j % tile_size;
            s_A[j] = (row_a < m && col_a < k) ? A[row_a * k + col_a] : 0.0f;
        }
        for (int j = threadIdx.x; j < tile_size * tile_size; j += blockDim.x) {
            int row_b = i + j / tile_size;
            int col_b = blockIdx.x * tile_size + j % tile_size;
            s_B[j] = (row_b < k && col_b < n) ? B[row_b * n + col_b] : 0.0f;
        }
        __syncthreads();

        if (row < m && col < n) {
            for (int kk = 0; kk < tile_size; kk++) {
                acc += s_A[ty * tile_size + kk] * s_B[kk * tile_size + tx];
            }
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * n + col] = acc;
    }
}


// X, y, beta are device pointers
extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {
    cudaStream_t stream_1, stream_2;
    cudaStreamCreate(&stream_1);
    cudaStreamCreate(&stream_2);
    // printf("the streams created\n");

    float *d_buf;
    float *d_xt;
    float *d_xtx;
    float *d_xty;
    float *d_xtx_inv;
    size_t buf_size = (n_features * n_samples + n_features + n_features * n_features + n_features * n_features) * sizeof(float);
    cudaMallocAsync(&d_buf, buf_size, stream_1);
    d_xt = d_buf;
    d_xty = d_xt + n_features * n_samples;
    d_xtx = d_xty + n_features;
    d_xtx_inv = d_xtx + n_features * n_features;

    cudaEvent_t trans_event, xty_event, eye_init_event;
    cudaEventCreate(&trans_event);
    cudaEventCreate(&xty_event);
    cudaEventCreate(&eye_init_event);

    // X^T -> (n_features, n_samples)
    constexpr int tile_size = 32;
    dim3 block(tile_size, tile_size);
    dim3 grid(div_up(n_features, tile_size), div_up(n_samples, tile_size));
    mat_transpose_kernel<tile_size><<<grid, block, 0, stream_1>>>(X, d_xt, n_samples, n_features);
    cudaGetLastError();
    cudaEventRecord(trans_event, stream_1);

    // init d_xtx_inv to identity matrix
    init_identity_matrix_kernel<<<1, 256, 0, stream_2>>>(d_xtx_inv, n_features);
    cudaGetLastError();
    cudaEventRecord(eye_init_event, stream_2);

    // xtx * y -> (n_features,)
    cudaStreamWaitEvent(stream_2, trans_event);
    mv_kernel<<<n_features, 128, 0, stream_2>>>(d_xt, y, d_xty, n_samples);
    cudaGetLastError();
    cudaEventRecord(xty_event, stream_2);

    // X^T * X -> (n_features, n_features)
    block = dim3(tile_size * tile_size);
    grid = dim3(div_up(n_features, tile_size), div_up(n_features, tile_size));
    matrix_muliply_kernel<tile_size><<<grid, block, 0, stream_1>>>(d_xt, X, d_xtx, n_features, n_features, n_samples);
    cudaGetLastError();

    // (X^T * X)^-1
    cudaStreamWaitEvent(stream_1, eye_init_event);
    gaussian_elimination_kernel<<<1, 1024, 0, stream_1>>>(d_xtx, d_xtx_inv, n_features);
    cudaGetLastError();

    // (X^T * X)^-1 * (X^T * y) -> (n_features,)
    cudaStreamWaitEvent(stream_1, xty_event);
    mv_kernel<<<n_features, 128, 0, stream_1>>>(d_xtx_inv, d_xty, beta, n_features);
    cudaGetLastError();
    cudaStreamSynchronize(stream_1);

    // free
    cudaEventDestroy(trans_event);
    cudaEventDestroy(xty_event);
    cudaEventDestroy(eye_init_event);
    cudaFreeAsync(d_buf, stream_1);
    cudaStreamDestroy(stream_1);
    cudaStreamDestroy(stream_2);
}
