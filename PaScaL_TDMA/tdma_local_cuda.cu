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

// Local Thomas — 6 shared buffers (bx+1, by); pipeline keeps the (j-1) row's
// c/d in shared so the forward sweep only loads the new row from global.
// Mirrors PaScaL_TDMA_F tdmas_cuda.f90 tdma_many_cuda.
__global__ __launch_bounds__(PASCAL_TDMA_MAX_BLOCK)
void tdma_many_kernel(double* __restrict__ A,
                      double* __restrict__ B,
                      double* __restrict__ C,
                      double* __restrict__ D,
                      int n_sys, int n_row) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockDim.x;
    const int by = blockDim.y;
    int i = blockIdx.x * (bx * by) + ty * bx + tx;
    if (i >= n_sys) return;

    const int slots = (bx + 1) * by;
    extern __shared__ double smem[];
    double* a1 = smem + 0 * slots;
    double* b1 = smem + 1 * slots;
    double* c0 = smem + 2 * slots;
    double* c1 = smem + 3 * slots;
    double* d0 = smem + 4 * slots;
    double* d1 = smem + 5 * slots;
    const int tj = ty * (bx + 1) + tx;

    b1[tj] = B[i];
    c1[tj] = C[i];
    d1[tj] = D[i];
    d1[tj] /= b1[tj];
    c1[tj] /= b1[tj];
    C[i] = c1[tj];
    D[i] = d1[tj];

    for (int j = 1; j < n_row; ++j) {
        std::size_t off = (std::size_t)j * n_sys + i;
        c0[tj] = c1[tj];
        d0[tj] = d1[tj];
        a1[tj] = A[off];
        b1[tj] = B[off];
        c1[tj] = C[off];
        d1[tj] = D[off];
        double r = 1.0 / (b1[tj] - a1[tj] * c0[tj]);
        d1[tj] = r * (d1[tj] - a1[tj] * d0[tj]);
        c1[tj] = r * c1[tj];
        C[off] = c1[tj];
        D[off] = d1[tj];
    }

    for (int j = n_row - 2; j >= 0; --j) {
        std::size_t off = (std::size_t)j * n_sys + i;
        c0[tj] = C[off];
        d0[tj] = D[off];
        d0[tj] = d0[tj] - c0[tj] * d1[tj];
        d1[tj] = d0[tj];
        D[off] = d0[tj];
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

// Modified Thomas pipeline — 9 shared buffers (bx+1, by); +1 padding to
// avoid 32-way bank conflicts. Mirrors PaScaL_TDMA_F tdmas_cuda.f90.
__global__ __launch_bounds__(PASCAL_TDMA_MAX_BLOCK)
void modified_thomas_kernel(double* __restrict__ A,
                            double* __restrict__ B,
                            double* __restrict__ C,
                            double* __restrict__ D,
                            double* __restrict__ A_rd,
                            double* __restrict__ B_rd,
                            double* __restrict__ C_rd,
                            double* __restrict__ D_rd,
                            int n_sys, int n_row) {
    int i = linear_sys(n_sys);
    if (i >= n_sys) return;

    const int slots = (blockDim.x + 1) * blockDim.y;
    const int tj    = threadIdx.y * (blockDim.x + 1) + threadIdx.x;
    extern __shared__ double smem[];
    double* a0 = smem + 0 * slots;
    double* b0 = smem + 1 * slots;
    double* c0 = smem + 2 * slots;
    double* d0 = smem + 3 * slots;
    double* a1 = smem + 4 * slots;
    double* b1 = smem + 5 * slots;
    double* c1 = smem + 6 * slots;
    double* d1 = smem + 7 * slots;
    double* r0 = smem + 8 * slots;

    a0[tj] = A[i];
    b0[tj] = B[i];
    c0[tj] = C[i];
    d0[tj] = D[i];
    a0[tj] /= b0[tj];
    c0[tj] /= b0[tj];
    d0[tj] /= b0[tj];
    A[i] = a0[tj];
    C[i] = c0[tj];
    D[i] = d0[tj];

    std::size_t off1 = (std::size_t)1 * n_sys + i;
    a1[tj] = A[off1];
    b1[tj] = B[off1];
    c1[tj] = C[off1];
    d1[tj] = D[off1];
    a1[tj] /= b1[tj];
    c1[tj] /= b1[tj];
    d1[tj] /= b1[tj];
    A[off1] = a1[tj];
    C[off1] = c1[tj];
    D[off1] = d1[tj];

    for (int j = 2; j < n_row; ++j) {
        a0[tj] = a1[tj];
        c0[tj] = c1[tj];
        d0[tj] = d1[tj];

        std::size_t off = (std::size_t)j * n_sys + i;
        a1[tj] = A[off];
        b1[tj] = B[off];
        c1[tj] = C[off];
        d1[tj] = D[off];

        r0[tj] =  1.0 / (b1[tj] - a1[tj] * c0[tj]);
        d1[tj] =  r0[tj] * (d1[tj] - a1[tj] * d0[tj]);
        c1[tj] =  r0[tj] * c1[tj];
        a1[tj] = -r0[tj] * a1[tj] * a0[tj];

        A[off] = a1[tj];
        C[off] = c1[tj];
        D[off] = d1[tj];
    }

    double a_last = a1[tj], c_last = c1[tj], d_last = d1[tj];

    {
        std::size_t off_nm2 = (std::size_t)(n_row - 2) * n_sys + i;
        a1[tj] = A[off_nm2];
        c1[tj] = C[off_nm2];
        d1[tj] = D[off_nm2];
    }

    for (int j = n_row - 3; j >= 1; --j) {
        std::size_t off = (std::size_t)j * n_sys + i;
        a0[tj] = A[off];
        c0[tj] = C[off];
        d0[tj] = D[off];

        d0[tj] = d0[tj] - c0[tj] * d1[tj];
        a0[tj] = a0[tj] - c0[tj] * a1[tj];
        c0[tj] = -c0[tj] * c1[tj];

        a1[tj] = a0[tj];
        c1[tj] = c0[tj];
        d1[tj] = d0[tj];

        A[off] = a0[tj];
        C[off] = c0[tj];
        D[off] = d0[tj];
    }

    {
        a0[tj] = A[i];
        c0[tj] = C[i];
        d0[tj] = D[i];

        r0[tj] = 1.0 / (1.0 - a1[tj] * c0[tj]);
        d0[tj] =  r0[tj] * (d0[tj] - c0[tj] * d1[tj]);
        a0[tj] =  r0[tj] * a0[tj];
        c0[tj] = -r0[tj] * c0[tj] * c1[tj];

        D[i] = d0[tj];
        A[i] = a0[tj];
        C[i] = c0[tj];

        A_rd[i] = a0[tj];
        B_rd[i] = 1.0;
        C_rd[i] = c0[tj];
        D_rd[i] = d0[tj];
    }

    std::size_t boff = (std::size_t)1 * n_sys + i;
    A_rd[boff] = a_last;
    B_rd[boff] = 1.0;
    C_rd[boff] = c_last;
    D_rd[boff] = d_last;
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
    std::size_t smem_bytes = 6 * (std::size_t)(block_x + 1)
                               * (std::size_t)block_y * sizeof(double);
    tdma_many_kernel<<<grid, block, smem_bytes, stream>>>(
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
    std::size_t smem_bytes = 9 * (std::size_t)(block_x + 1)
                               * (std::size_t)block_y * sizeof(double);
    modified_thomas_kernel<<<grid, block, smem_bytes, stream>>>(
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
