#include "tdma_local_cuda.cuh"

#include <cstddef>

// Layout convention: row-major [n_row × n_sys].
//   element (j, i) is at offset j*n_sys + i.
// Each CUDA thread owns one independent system (column i), and walks the rows
// sequentially. With this layout, threads in the same warp access consecutive
// memory at every step (perfectly coalesced).

namespace {

inline dim3 grid_for(int n_sys) {
    return dim3((n_sys + PASCAL_TDMA_BLOCK_X - 1) / PASCAL_TDMA_BLOCK_X);
}

// ---------------------------------------------------------------------------
//  Standard Thomas algorithm — non-cyclic, single-rank (or local-reduced).
//  Mirrors tdma_many in tdma_local.cpp.
// ---------------------------------------------------------------------------
__global__ void tdma_many_kernel(double* __restrict__ A,
                                 double* __restrict__ B,
                                 double* __restrict__ C,
                                 double* __restrict__ D,
                                 int n_sys, int n_row) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;

    // Row 0: normalize by B[0]
    double inv_b = 1.0 / B[i];
    double c_prev = C[i] * inv_b;
    double d_prev = D[i] * inv_b;
    C[i] = c_prev;
    D[i] = d_prev;

    // Forward elimination: rows 1 .. n_row-1
    for (int j = 1; j < n_row; ++j) {
        std::size_t off = (std::size_t)j * n_sys + i;
        double aj = A[off];
        double bj = B[off];
        double cj = C[off];
        double dj = D[off];
        double r  = 1.0 / (bj - aj * c_prev);
        d_prev = r * (dj - aj * d_prev);
        c_prev = r * cj;
        C[off] = c_prev;
        D[off] = d_prev;
    }

    // d_prev now equals D[n_row-1]. Backward substitution: rows n_row-2 .. 0
    for (int j = n_row - 2; j >= 0; --j) {
        std::size_t off = (std::size_t)j * n_sys + i;
        double cj = C[off];
        double dj = D[off];
        dj = dj - cj * d_prev;
        d_prev = dj;
        D[off] = dj;
    }
}

// ---------------------------------------------------------------------------
//  Cyclic Thomas algorithm via Sherman-Morrison.
//  Mirrors tdma_cyclic_many in tdma_local.cpp. d_E is workspace [n_row × n_sys].
// ---------------------------------------------------------------------------
__global__ void tdma_cyclic_many_kernel(double* __restrict__ A,
                                        double* __restrict__ B,
                                        double* __restrict__ C,
                                        double* __restrict__ D,
                                        double* __restrict__ E,
                                        int n_sys, int n_row) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;

    // Initialize E: zeros except E[1] = -A[1], E[n_row-1] = -C[n_row-1]
    for (int j = 0; j < n_row; ++j) E[(std::size_t)j * n_sys + i] = 0.0;
    E[(std::size_t)1 * n_sys + i]            = -A[(std::size_t)1 * n_sys + i];
    E[(std::size_t)(n_row - 1) * n_sys + i]  = -C[(std::size_t)(n_row - 1) * n_sys + i];

    // Preprocess row 1
    {
        std::size_t off = (std::size_t)1 * n_sys + i;
        double inv_b = 1.0 / B[off];
        D[off] *= inv_b;
        E[off] *= inv_b;
        C[off] *= inv_b;
    }

    // Forward elimination: rows 2 .. n_row-1
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

    // Backward substitution: rows n_row-2 .. 1
    for (int j = n_row - 2; j >= 1; --j) {
        std::size_t off  = (std::size_t)j * n_sys + i;
        std::size_t offp = (std::size_t)(j + 1) * n_sys + i;
        double cj = C[off];
        D[off] -= cj * D[offp];
        E[off] -= cj * E[offp];
    }

    // Solve for D[0]
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

    // Back-substitute D[0] into rows 1..n_row-1
    double d0 = D[i];
    for (int j = 1; j < n_row; ++j) {
        std::size_t off = (std::size_t)j * n_sys + i;
        D[off] += d0 * E[off];
    }
}

// ---------------------------------------------------------------------------
//  Modified Thomas algorithm: forward + backward sweep that produces a
//  2-row reduced system at boundary rows {0, n_row-1} for each system.
//  Mirrors steps 1..3 of PaScaLTDMAMany::solve in pascal_tdma_many.cpp.
// ---------------------------------------------------------------------------
__global__ void modified_thomas_kernel(double* __restrict__ A,
                                       double* __restrict__ B,
                                       double* __restrict__ C,
                                       double* __restrict__ D,
                                       double* __restrict__ A_rd,
                                       double* __restrict__ B_rd,
                                       double* __restrict__ C_rd,
                                       double* __restrict__ D_rd,
                                       int n_sys, int n_row) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;

    // Row 0 and 1: normalize by their B
    {
        double r0 = 1.0 / B[i];
        A[i] *= r0; C[i] *= r0; D[i] *= r0;
    }
    {
        std::size_t off1 = (std::size_t)1 * n_sys + i;
        double r1 = 1.0 / B[off1];
        A[off1] *= r1; C[off1] *= r1; D[off1] *= r1;
    }

    // Forward elimination: rows 2 .. n_row-1
    //   aj = -inv * aj * a[j-1]   (cumulative product of "a" coupling to row 0)
    for (int j = 2; j < n_row; ++j) {
        std::size_t off  = (std::size_t)j * n_sys + i;
        std::size_t offm = (std::size_t)(j - 1) * n_sys + i;
        double aj = A[off];
        double bj = B[off];
        double cj = C[off];
        double dj = D[off];
        double inv = 1.0 / (bj - aj * C[offm]);
        D[off] =  inv * (dj - aj * D[offm]);
        C[off] =  inv * cj;
        A[off] = -inv * aj * A[offm];
    }

    // Backward sweep: rows n_row-3 .. 1 — eliminate upper coupling using row n_row-1
    for (int j = n_row - 3; j >= 1; --j) {
        std::size_t off  = (std::size_t)j * n_sys + i;
        std::size_t offp = (std::size_t)(j + 1) * n_sys + i;
        double cj = C[off];
        D[off] -= cj * D[offp];
        A[off] -= cj * A[offp];
        C[off]  = -cj * C[offp];
    }

    // Reduce row 0 using row 1
    double a1   = A[(std::size_t)1 * n_sys + i];
    double c0_v = C[i];
    double d1   = D[(std::size_t)1 * n_sys + i];

    double r = 1.0 / (1.0 - a1 * c0_v);
    double d0_v = r * (D[i] - c0_v * d1);
    double a0_v = r * A[i];
    double cN0  = -r * c0_v * C[(std::size_t)1 * n_sys + i];

    D[i] = d0_v;
    A[i] = a0_v;
    C[i] = cN0;

    // Pack reduced system [2 × n_sys]: row 0 = local row 0, row 1 = local row n_row-1
    std::size_t boff = (std::size_t)1 * n_sys + i;
    std::size_t lastN = (std::size_t)(n_row - 1) * n_sys + i;

    A_rd[i]    = a0_v;
    B_rd[i]    = 1.0;
    C_rd[i]    = cN0;
    D_rd[i]    = d0_v;

    A_rd[boff] = A[lastN];
    B_rd[boff] = 1.0;
    C_rd[boff] = C[lastN];
    D_rd[boff] = D[lastN];
}

// ---------------------------------------------------------------------------
//  Final update step: with the reduced solution placed in D_rd[0..1, :],
//  fill D[0]=D_rd0, D[n_row-1]=D_rd1, then for j in 1..n_row-2 do
//      D[j] -= A[j]*D_rd0 + C[j]*D_rd1
//  Mirrors step 7 of PaScaLTDMAMany::solve.
// ---------------------------------------------------------------------------
__global__ void update_solution_kernel(const double* __restrict__ A,
                                       const double* __restrict__ C,
                                       double* __restrict__ D,
                                       const double* __restrict__ D_rd,
                                       int n_sys, int n_row) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
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

// ---------------------------------------------------------------------------
//  Host-side launchers
// ---------------------------------------------------------------------------

void tdma_many_cuda(double* d_A, double* d_B, double* d_C, double* d_D,
                    int n_sys, int n_row, cudaStream_t stream) {
    dim3 block(PASCAL_TDMA_BLOCK_X);
    dim3 grid = grid_for(n_sys);
    tdma_many_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, d_D, n_sys, n_row);
}

void tdma_cyclic_many_cuda(double* d_A, double* d_B, double* d_C, double* d_D,
                           double* d_E, int n_sys, int n_row, cudaStream_t stream) {
    dim3 block(PASCAL_TDMA_BLOCK_X);
    dim3 grid = grid_for(n_sys);
    tdma_cyclic_many_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, d_D, d_E, n_sys, n_row);
}

void pascal_tdma_modified_thomas_cuda(double* d_A, double* d_B, double* d_C, double* d_D,
                                      double* d_A_rd, double* d_B_rd,
                                      double* d_C_rd, double* d_D_rd,
                                      int n_sys, int n_row, cudaStream_t stream) {
    dim3 block(PASCAL_TDMA_BLOCK_X);
    dim3 grid = grid_for(n_sys);
    modified_thomas_kernel<<<grid, block, 0, stream>>>(
        d_A, d_B, d_C, d_D, d_A_rd, d_B_rd, d_C_rd, d_D_rd, n_sys, n_row);
}

void pascal_tdma_update_solution_cuda(double* d_A, double* d_C, double* d_D,
                                      const double* d_D_rd,
                                      int n_sys, int n_row, cudaStream_t stream) {
    dim3 block(PASCAL_TDMA_BLOCK_X);
    dim3 grid = grid_for(n_sys);
    update_solution_kernel<<<grid, block, 0, stream>>>(d_A, d_C, d_D, d_D_rd, n_sys, n_row);
}
