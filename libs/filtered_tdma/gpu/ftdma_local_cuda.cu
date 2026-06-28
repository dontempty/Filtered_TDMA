#include "ftdma_local_cuda.cuh"

#include <cstddef>
#include <cstdlib>

namespace {

constexpr int FTDMA_MAX_BLOCK = 512;

inline dim3 grid_for(int n_sys, int block_total) {
    return dim3((n_sys + block_total - 1) / block_total);
}

__device__ __forceinline__ int linear_tid() {
    return threadIdx.y * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ int block_total() {
    return blockDim.x * blockDim.y;
}

__device__ __forceinline__ int linear_sys(int n_sys) {
    return blockIdx.x * block_total() + linear_tid();
}

// Local Thomas algorithm — register-resident scalars.
// Layout: row-major [n_row × n_sys].
__global__ __launch_bounds__(256)
void ftdma_many_kernel(double* __restrict__ A,
                       double* __restrict__ B,
                       double* __restrict__ C,
                       double* __restrict__ D,
                       int n_sys, int n_row) {
    int i = blockIdx.x * blockDim.x * blockDim.y
          + threadIdx.y * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;

    double b1 = B[i], c1 = C[i], d1 = D[i];
    d1 /= b1;  c1 /= b1;
    C[i] = c1;  D[i] = d1;

    for (int j = 1; j < n_row; ++j) {
        double c0 = c1, d0 = d1;
        std::size_t off = (std::size_t)j * n_sys + i;
        double a1 = A[off];
        b1 = B[off];  c1 = C[off];  d1 = D[off];
        double r = 1.0 / (b1 - a1 * c0);
        d1 = r * (d1 - a1 * d0);
        c1 = r * c1;
        C[off] = c1;  D[off] = d1;
    }

    double d_prev = d1;
    for (int j = n_row - 2; j >= 0; --j) {
        std::size_t off = (std::size_t)j * n_sys + i;
        double d = D[off] - C[off] * d_prev;
        D[off] = d;  d_prev = d;
    }
}

// Cyclic Thomas via Sherman-Morrison. d_E is [n_row × n_sys] workspace.
__global__ __launch_bounds__(FTDMA_MAX_BLOCK)
void ftdma_cyclic_many_kernel(double* __restrict__ A,
                              double* __restrict__ B,
                              double* __restrict__ C,
                              double* __restrict__ D,
                              double* __restrict__ E,
                              int n_sys, int n_row) {
    int i = linear_sys(n_sys);
    if (i >= n_sys) return;

    for (int j = 0; j < n_row; ++j) E[(std::size_t)j * n_sys + i] = 0.0;
    E[(std::size_t)1 * n_sys + i]            = -A[(std::size_t)1 * n_sys + i];
    E[(std::size_t)(n_row - 1) * n_sys + i]  = -C[(std::size_t)(n_row - 1) * n_sys + i];

    {
        std::size_t off = (std::size_t)1 * n_sys + i;
        double inv_b = 1.0 / B[off];
        D[off] *= inv_b;
        E[off] *= inv_b;
        C[off] *= inv_b;
    }

    for (int j = 2; j < n_row; ++j) {
        std::size_t off  = (std::size_t)j * n_sys + i;
        std::size_t offm = (std::size_t)(j - 1) * n_sys + i;
        double aj = A[off];
        double bj = B[off];
        double cj = C[off];
        double dj = D[off];
        double ej = E[off];
        double r  = 1.0 / (bj - aj * C[offm]);
        D[off] = r * (dj - aj * D[offm]);
        E[off] = r * (ej - aj * E[offm]);
        C[off] = r * cj;
    }

    for (int j = n_row - 2; j >= 1; --j) {
        std::size_t off  = (std::size_t)j * n_sys + i;
        std::size_t offp = (std::size_t)(j + 1) * n_sys + i;
        double cj = C[off];
        D[off] -= cj * D[offp];
        E[off] -= cj * E[offp];
    }

    {
        double a0   = A[i];
        double b0   = B[i];
        double c0   = C[i];
        double d0   = D[i];
        double d1   = D[(std::size_t)1 * n_sys + i];
        double dN1  = D[(std::size_t)(n_row - 1) * n_sys + i];
        double e1   = E[(std::size_t)1 * n_sys + i];
        double eN1  = E[(std::size_t)(n_row - 1) * n_sys + i];
        D[i] = (d0 - a0 * dN1 - c0 * d1) / (b0 + a0 * eN1 + c0 * e1);
    }

    double d0 = D[i];
    for (int j = 1; j < n_row; ++j) {
        std::size_t off = (std::size_t)j * n_sys + i;
        D[off] += d0 * E[off];
    }
}

} // anonymous namespace

void ftdma_many_cuda(double* d_A, double* d_B, double* d_C, double* d_D,
                     int n_sys, int n_row,
                     int block_x, int block_y, cudaStream_t stream) {
    int bt = block_x * block_y;
    dim3 block(block_x, block_y);
    dim3 grid = grid_for(n_sys, bt);
    ftdma_many_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, d_D, n_sys, n_row);
}

void ftdma_cyclic_many_cuda(double* d_A, double* d_B, double* d_C, double* d_D,
                            double* d_E, int n_sys, int n_row,
                            int block_x, int block_y, cudaStream_t stream) {
    int bt = block_x * block_y;
    dim3 block(block_x, block_y);
    dim3 grid = grid_for(n_sys, bt);
    ftdma_cyclic_many_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, d_D, d_E, n_sys, n_row);
}
