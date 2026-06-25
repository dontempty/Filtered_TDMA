#include "tdma_local_cuda.cuh"

#include <cstddef>
#include <cstdlib>

// Layout: row-major [n_row × n_sys]. Each thread owns one column i; warps
// access consecutive systems → coalesced.

namespace {

constexpr int PASCAL_TDMA_MAX_BLOCK = 512;

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

// Local Thomas — register-resident scalars, mirrors Fortran tdma_many_cuda.
// No shared memory: each thread owns one column and keeps the sliding-window
// (c0, d0 from row j-1) in registers, identical to Fortran local variables.
__global__ __launch_bounds__(256)
void tdma_many_kernel(double* __restrict__ A,
                      double* __restrict__ B,
                      double* __restrict__ C,
                      double* __restrict__ D,
                      int n_sys, int n_row) {
    int i = blockIdx.x * blockDim.x * blockDim.y
          + threadIdx.y * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;

    // Row 0: normalize (b is never stored back; only c and d needed later)
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
__global__ __launch_bounds__(PASCAL_TDMA_MAX_BLOCK)
void tdma_cyclic_many_kernel(double* __restrict__ A,
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

// Modified Thomas — register-resident scalars, mirrors Fortran tdma_modified_cuda.
// No shared memory: a0/b0/c0/d0 hold row j-1, a1/b1/c1/d1 hold row j.
// The _sh suffix in Fortran is just a naming convention; they are plain
// local scalars (registers), not Fortran shared-memory variables.
__global__ __launch_bounds__(256)
void modified_thomas_kernel(double* __restrict__ A,
                            double* __restrict__ B,
                            double* __restrict__ C,
                            double* __restrict__ D,
                            double* __restrict__ A_rd,
                            double* __restrict__ B_rd,
                            double* __restrict__ C_rd,
                            double* __restrict__ D_rd,
                            int n_sys, int n_row) {
    int i = blockIdx.x * blockDim.x * blockDim.y
          + threadIdx.y * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;

    // Row 0
    double a0 = A[i], b0 = B[i], c0 = C[i], d0 = D[i];
    a0 /= b0;  c0 /= b0;  d0 /= b0;
    A[i] = a0;  C[i] = c0;  D[i] = d0;

    // Row 1
    std::size_t off1 = (std::size_t)1 * n_sys + i;
    double a1 = A[off1], b1 = B[off1], c1 = C[off1], d1 = D[off1];
    a1 /= b1;  c1 /= b1;  d1 /= b1;
    A[off1] = a1;  C[off1] = c1;  D[off1] = d1;

    // Forward elimination rows 2..n_row-1
    for (int j = 2; j < n_row; ++j) {
        a0 = a1;  c0 = c1;  d0 = d1;
        std::size_t off = (std::size_t)j * n_sys + i;
        a1 = A[off];  b1 = B[off];  c1 = C[off];  d1 = D[off];
        double r0 = 1.0 / (b1 - a1 * c0);
        d1 =  r0 * (d1 - a1 * d0);
        c1 =  r0 * c1;
        a1 = -r0 * a1 * a0;
        A[off] = a1;  C[off] = c1;  D[off] = d1;
    }

    // Reduced row 1 (last row of forward sweep) — still in registers
    {
        std::size_t boff = (std::size_t)1 * n_sys + i;
        A_rd[boff] = a1;  B_rd[boff] = 1.0;  C_rd[boff] = c1;  D_rd[boff] = d1;
    }

    // Backward sweep — load row n_row-2 into (a1,c1,d1) sliding window
    {
        std::size_t off_nm2 = (std::size_t)(n_row - 2) * n_sys + i;
        a1 = A[off_nm2];  c1 = C[off_nm2];  d1 = D[off_nm2];
    }

    for (int j = n_row - 3; j >= 1; --j) {
        std::size_t off = (std::size_t)j * n_sys + i;
        a0 = A[off];  c0 = C[off];  d0 = D[off];
        d0 = d0 - c0 * d1;
        a0 = a0 - c0 * a1;
        c0 = -c0 * c1;
        a1 = a0;  c1 = c0;  d1 = d0;
        A[off] = a0;  C[off] = c0;  D[off] = d0;
    }

    // Final 2×2 reduction at row 0
    {
        a0 = A[i];  c0 = C[i];  d0 = D[i];
        double r0 = 1.0 / (1.0 - a1 * c0);
        d0 =  r0 * (d0 - c0 * d1);
        a0 =  r0 * a0;
        c0 = -r0 * c0 * c1;
        D[i] = d0;  A[i] = a0;  C[i] = c0;
        A_rd[i] = a0;  B_rd[i] = 1.0;  C_rd[i] = c0;  D_rd[i] = d0;
    }
}

__global__ __launch_bounds__(PASCAL_TDMA_MAX_BLOCK)
void update_solution_kernel(const double* __restrict__ A,
                            const double* __restrict__ C,
                            double* __restrict__ D,
                            const double* __restrict__ D_rd,
                            int n_sys, int n_row) {
    int i = linear_sys(n_sys);
    if (i >= n_sys) return;

    double d0 = D_rd[i];
    double dN = D_rd[(std::size_t)1 * n_sys + i];

    D[i] = d0;
    D[(std::size_t)(n_row - 1) * n_sys + i] = dN;

    for (int j = 1; j < n_row - 1; ++j) {
        std::size_t off = (std::size_t)j * n_sys + i;
        D[off] -= A[off] * d0 + C[off] * dN;
    }
}

} // anonymous namespace

void tdma_many_cuda(double* d_A, double* d_B, double* d_C, double* d_D,
                    int n_sys, int n_row,
                    int block_x, int block_y, cudaStream_t stream) {
    int block_total_v = block_x * block_y;
    dim3 block(block_x, block_y);
    dim3 grid = grid_for(n_sys, block_total_v);
    tdma_many_kernel<<<grid, block, 0, stream>>>(
        d_A, d_B, d_C, d_D, n_sys, n_row);
}

void tdma_cyclic_many_cuda(double* d_A, double* d_B, double* d_C, double* d_D,
                           double* d_E, int n_sys, int n_row,
                           int block_x, int block_y, cudaStream_t stream) {
    int block_total_v = block_x * block_y;
    dim3 block(block_x, block_y);
    dim3 grid = grid_for(n_sys, block_total_v);
    tdma_cyclic_many_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, d_D, d_E, n_sys, n_row);
}

void pascal_tdma_modified_thomas_cuda(double* d_A, double* d_B, double* d_C, double* d_D,
                                      double* d_A_rd, double* d_B_rd,
                                      double* d_C_rd, double* d_D_rd,
                                      int n_sys, int n_row,
                                      int block_x, int block_y, cudaStream_t stream) {
    int block_total_v = block_x * block_y;
    dim3 block(block_x, block_y);
    dim3 grid = grid_for(n_sys, block_total_v);
    modified_thomas_kernel<<<grid, block, 0, stream>>>(
        d_A, d_B, d_C, d_D, d_A_rd, d_B_rd, d_C_rd, d_D_rd, n_sys, n_row);
}

void pascal_tdma_update_solution_cuda(double* d_A, double* d_C, double* d_D,
                                      const double* d_D_rd,
                                      int n_sys, int n_row,
                                      int block_x, int block_y, cudaStream_t stream) {
    int block_total_v = block_x * block_y;
    dim3 block(block_x, block_y);
    dim3 grid = grid_for(n_sys, block_total_v);
    update_solution_kernel<<<grid, block, 0, stream>>>(d_A, d_C, d_D, d_D_rd, n_sys, n_row);
}
