#include "filtered_tdma_cuda.hpp"
#include "tdma_local_cuda.cuh"

#include <cuda_runtime.h>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <cstdlib>

// =============================================================================
//  CUDA helpers
// =============================================================================
#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t err = (x);                                                 \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "[CUDA] %s:%d %s -> %s\n", __FILE__, __LINE__,\
                         #x, cudaGetErrorString(err));                         \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

static constexpr int BLOCK_SYS = 256;

static inline dim3 grid1D(int n) { return dim3((n + BLOCK_SYS - 1) / BLOCK_SYS); }

// Warp-level max reduction using shuffle (no __syncthreads needed within a warp).
// Caller must be in a warp of 32 active threads (tid < 32).
__device__ inline double warp_max(double v) {
    for (int s = 16; s > 0; s >>= 1)
        v = fmax(v, __shfl_down_sync(0xffffffffu, v, s));
    return v;
}

// Hybrid shared-memory + warp-shuffle tree reduction for three doubles.
// Requires s_a/s_b/s_c to be BLOCK_SYS-element shared arrays pre-filled.
// After the call only tid==0 holds the final maxima (in s_a[0]/s_b[0]/s_c[0]).
__device__ inline void reduce3_max(double* s_a, double* s_b, double* s_c, int tid) {
    // Shared-memory phase: 256 → 32
    for (int s = BLOCK_SYS / 2; s >= 32; s >>= 1) {
        if (tid < s) {
            s_a[tid] = fmax(s_a[tid], s_a[tid + s]);
            s_b[tid] = fmax(s_b[tid], s_b[tid + s]);
            s_c[tid] = fmax(s_c[tid], s_c[tid + s]);
        }
        __syncthreads();
    }
    // Warp-shuffle phase: 32 → 1 (no sync needed)
    if (tid < 32) {
        double a = warp_max(s_a[tid]);
        double b = warp_max(s_b[tid]);
        double c = warp_max(s_c[tid]);
        if (tid == 0) { s_a[0] = a; s_b[0] = b; s_c[0] = c; }
    }
    __syncthreads();  // make tid==0 result visible to all threads
}

// =============================================================================
//  Kernels  (one thread per "system" index i ∈ [0, n_sys))
// =============================================================================

// Compute per-row A_rho[k] = |A[k*n_sys] / B[k*n_sys]|,
//             C_rho[k] = |C[k*n_sys] / B[k*n_sys]|
// Launched with grid1D(n_row), one thread per row (k along x).
__global__ void k_set_rho(double* __restrict__ d_A_rho,
                          double* __restrict__ d_C_rho,
                          const double* __restrict__ d_A,
                          const double* __restrict__ d_B,
                          const double* __restrict__ d_C,
                          int n_sys, int n_row) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_row + 1) return;
    if (k < n_row) {
        // Each thread reads one element from a column-0 slice of a row-major
        // matrix: stride = n_sys doubles.  __ldg routes through the read-only
        // (texture) cache, which handles non-unit strides gracefully.
        double bk = __ldg(&d_B[(std::size_t)k * n_sys]);
        if (bk != 0.0) {
            d_A_rho[k] = fabs(__ldg(&d_A[(std::size_t)k * n_sys]) / bk);
            d_C_rho[k] = fabs(__ldg(&d_C[(std::size_t)k * n_sys]) / bk);
        } else {
            d_A_rho[k] = 0.0;
            d_C_rho[k] = 0.0;
        }
    } else {
        // trailing slot stays zero (matches CPU FilteredTDMA::A_rho_.size()=n_row+1).
        d_A_rho[k] = 0.0;
        d_C_rho[k] = 0.0;
    }
}

// Merged forward elimination: row 0, row 1 normalize + forward elim rows
// 2..n_row-1, all in a single kernel. Each thread owns one system (column i)
// and walks the rows internally — same structure as PaScaL_TDMA's
// modified_thomas_kernel. Replaces (1 + n_row-2) individual kernel launches.
__global__ __launch_bounds__(BLOCK_SYS, 4)
void k_fwd_pass(double* A, double* B, double* C, double* D,
                           int n_sys, int n_row) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;

    double r0 = 1.0 / B[i];
    A[i] *= r0; C[i] *= r0; D[i] *= r0;

    std::size_t off1 = (std::size_t)1 * n_sys + i;
    double r1 = 1.0 / B[off1];
    A[off1] *= r1; C[off1] *= r1; D[off1] *= r1;

    for (int j = 2; j < n_row; ++j) {
        std::size_t off_j  = (std::size_t)j * n_sys + i;
        std::size_t off_jm = (std::size_t)(j - 1) * n_sys + i;
        double Aj = A[off_j],  Bj = B[off_j],  Cj = C[off_j],  Dj = D[off_j];
        double Ajm = A[off_jm], Cjm = C[off_jm], Djm = D[off_jm];
        double inv = 1.0 / (Bj - Aj * Cjm);
        D[off_j] =  inv * (Dj - Aj * Djm);
        C[off_j] =  inv * Cj;
        A[off_j] = -inv * Aj * Ajm;
    }
}

// Merged backward sweep: rows n_row-3 .. 1. Each thread walks the rows
// backward — single launch replaces the per-row k_bwd_step loop.
__global__ __launch_bounds__(BLOCK_SYS, 4)
void k_bwd_pass(double* A, double* C, double* D,
                int n_sys, int n_row) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;

    for (int j = n_row - 3; j >= 1; --j) {
        std::size_t off_j  = (std::size_t)j * n_sys + i;
        std::size_t off_jp = (std::size_t)(j + 1) * n_sys + i;
        double Cj = C[off_j];
        double Djp = D[off_jp], Ajp = A[off_jp], Cjp = C[off_jp];
        D[off_j] -= Cj * Djp;
        A[off_j] -= Cj * Ajp;
        C[off_j]  = -Cj * Cjp;
    }
}

// v2 merged forward sweep: row 0,1 normalize, then full update (rows 2..J),
// then skip-A update (rows J+1..n_row-1). Single kernel — replaces
// k_fwd_norm_01 + per-row k_fwd_step + per-row k_fwd_step_skipA loops.
// J is read from device memory (broadcast via L2 cache, no D2H copy needed).
__global__ __launch_bounds__(BLOCK_SYS, 4)
void k_fwd_pass_v2(double* A, double* B, double* C, double* D,
                   int n_sys, int n_row, const int* __restrict__ d_J) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;
    const int J = __ldg(d_J);

    double r0 = 1.0 / B[i];
    A[i] *= r0; C[i] *= r0; D[i] *= r0;

    std::size_t off1 = (std::size_t)1 * n_sys + i;
    double r1 = 1.0 / B[off1];
    A[off1] *= r1; C[off1] *= r1; D[off1] *= r1;

    for (int j = 2; j <= J; ++j) {
        std::size_t off_j  = (std::size_t)j * n_sys + i;
        std::size_t off_jm = (std::size_t)(j - 1) * n_sys + i;
        double Aj = A[off_j],  Bj = B[off_j],  Cj = C[off_j],  Dj = D[off_j];
        double Ajm = A[off_jm], Cjm = C[off_jm], Djm = D[off_jm];
        double inv = 1.0 / (Bj - Aj * Cjm);
        D[off_j] =  inv * (Dj - Aj * Djm);
        C[off_j] =  inv * Cj;
        A[off_j] = -inv * Aj * Ajm;
    }
    for (int j = J + 1; j < n_row; ++j) {
        std::size_t off_j  = (std::size_t)j * n_sys + i;
        std::size_t off_jm = (std::size_t)(j - 1) * n_sys + i;
        double Aj = A[off_j],  Bj = B[off_j],  Cj = C[off_j],  Dj = D[off_j];
        double Cjm = C[off_jm], Djm = D[off_jm];
        double inv = 1.0 / (Bj - Aj * Cjm);
        D[off_j] = inv * (Dj - Aj * Djm);
        C[off_j] = inv * Cj;
    }
}

// v2 merged backward sweep — 3 phases (D-only over all, A-only up to J-1,
// C-multiply down to lo+1). Single kernel replaces per-row launches.
// J is read from device memory; lo is derived on-device (lo = (n_row-2) - J).
__global__ __launch_bounds__(BLOCK_SYS, 4)
void k_bwd_pass_v2(double* A, double* C, double* D,
                   int n_sys, int n_row, const int* __restrict__ d_J) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;
    const int J  = __ldg(d_J);
    const int lo = (n_row - 2) - J;

    for (int j = n_row - 3; j >= 1; --j) {
        std::size_t off_j  = (std::size_t)j * n_sys + i;
        std::size_t off_jp = (std::size_t)(j + 1) * n_sys + i;
        D[off_j] -= C[off_j] * D[off_jp];
    }
    if (J >= 2) {
        for (int j = J - 1; j >= 1; --j) {
            std::size_t off_j  = (std::size_t)j * n_sys + i;
            std::size_t off_jp = (std::size_t)(j + 1) * n_sys + i;
            A[off_j] -= C[off_j] * A[off_jp];
        }
    }
    for (int j = n_row - 3; j >= lo + 1; --j) {
        std::size_t off_j  = (std::size_t)j * n_sys + i;
        std::size_t off_jp = (std::size_t)(j + 1) * n_sys + i;
        C[off_j] *= -C[off_jp];
    }
}

// Merged final corrections. Applies the left correction for j in [1, J]
// and the right correction for j in [(n_row-1)-J, n_row-2] — single launch.
// J read from device memory (no D2H copy required).
__global__ __launch_bounds__(BLOCK_SYS, 4)
void k_final_pass(const double* A, const double* C, double* D,
                  const double* D0, const double* DN,
                  int n_sys, int n_row, const int* __restrict__ d_J) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;

    const int J = __ldg(d_J);
    double d0 = D0[i];
    double dn = DN[i];
    int jL = J < n_row - 2 ? J : n_row - 2;
    int jR = (n_row - 1) - J;
    if (jR < 1) jR = 1;

    for (int j = 1; j <= jL; ++j) {
        std::size_t off = (std::size_t)j * n_sys + i;
        D[off] -= A[off] * d0;
    }
    for (int j = jR; j < n_row - 1; ++j) {
        std::size_t off = (std::size_t)j * n_sys + i;
        D[off] -= C[off] * dn;
    }
}

// Pack reduced system — combine rows 0 and 1 (decouple row 0 from x_{N-1}).
__global__ void k_pack(double* A, double* C, double* D, int n_sys) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;
    double A1 = A[(std::size_t)1 * n_sys + i];
    double C0 = C[i];
    double D0 = D[i];
    double D1 = D[(std::size_t)1 * n_sys + i];
    double r = 1.0 / (1.0 - A1 * C0);
    D[i] = r * (D0 - C0 * D1);
    A[i] = r * A[i];
}

// Single-block GPU implementation of cal_J_v1: reduces A_rho/C_rho into rho,
// reduces |D0|/|DN| over n_sys into max_D0/max_DN, computes J using device
// math (sqrt/log). Writes the result to d_J. Replaces host-side reduction
// loops that triggered SIGILL on NVHPC nvc++ libm intrinsics after MPI.
__global__ void k_cal_J_v1(int* __restrict__ d_J,
                           const double* __restrict__ d_A_rho,
                           const double* __restrict__ d_C_rho,
                           const double* __restrict__ D0,
                           const double* __restrict__ DN,
                           int n_sys, int n_row, double eps) {
    __shared__ double s_rho[BLOCK_SYS];
    __shared__ double s_D0 [BLOCK_SYS];
    __shared__ double s_DN [BLOCK_SYS];
    int tid = threadIdx.x;

    // rho over k in [2, n_row-1) of d_A_rho/d_C_rho.
    double my_rho = 0.0;
    for (int k = 2 + tid; k < n_row - 1; k += blockDim.x)
        my_rho = fmax(my_rho, fmax(fabs(d_A_rho[k]), fabs(d_C_rho[k])));

    // |D0|, |DN| reductions over n_sys (coalesced reads).
    double my_D0 = 0.0, my_DN = 0.0;
    for (int i = tid; i < n_sys; i += blockDim.x) {
        my_D0 = fmax(my_D0, fabs(D0[i]));
        my_DN = fmax(my_DN, fabs(DN[i]));
    }

    s_rho[tid] = my_rho; s_D0[tid] = my_D0; s_DN[tid] = my_DN;
    __syncthreads();
    reduce3_max(s_rho, s_D0, s_DN, tid);

    if (tid == 0) {
        double rho = s_rho[0], max_D0 = s_D0[0], max_DN = s_DN[0];
        int J;
        if (rho == 0.0 || rho >= 0.5) {
            J = n_row - 1;
        } else {
            double lambda_p = (1.0 + sqrt(1.0 - 4.0 * rho * rho)) * 0.5;
            double q = rho / lambda_p;
            double K = (max_D0 + max_DN) * q / (1.0 - q);
            int Jcomp = (int)(log(eps / K) / log(q)) + 1;
            J = (Jcomp < n_row - 1) ? Jcomp : n_row - 1;
        }
        *d_J = J;
    }
}

// Single-block GPU implementation of cal_J_rhs_bound.
// Reduces max|d_D| over all n_row*n_sys elements, then computes
//   B = q(2+q) / ((1-q)(1-2*rho)) * max_b,  J = floor(log(eps/B)/log(q)) + 1.
__global__ void k_cal_J_rhs_bound(int* __restrict__ d_J,
                                   const double* __restrict__ d_A_rho,
                                   const double* __restrict__ d_C_rho,
                                   const double* __restrict__ d_D,
                                   int n_sys, int n_row, double eps) {
    __shared__ double s_rho[BLOCK_SYS];
    __shared__ double s_b  [BLOCK_SYS];
    int tid = threadIdx.x;

    double my_rho = 0.0;
    for (int k = 2 + tid; k < n_row - 1; k += blockDim.x)
        my_rho = fmax(my_rho, fmax(fabs(d_A_rho[k]), fabs(d_C_rho[k])));

    double my_b = 0.0;
    std::size_t total = (std::size_t)n_row * n_sys;
    for (std::size_t idx = (std::size_t)tid; idx < total; idx += blockDim.x)
        my_b = fmax(my_b, fabs(d_D[idx]));

    s_rho[tid] = my_rho; s_b[tid] = my_b;
    __syncthreads();
    // Two-array reduction: reuse reduce3_max with a dummy third array.
    // Use s_b as s_c placeholder (same pointer); a dummy shared slot is fine here.
    for (int s = BLOCK_SYS / 2; s >= 32; s >>= 1) {
        if (tid < s) {
            s_rho[tid] = fmax(s_rho[tid], s_rho[tid + s]);
            s_b  [tid] = fmax(s_b  [tid], s_b  [tid + s]);
        }
        __syncthreads();
    }
    if (tid < 32) {
        double r = warp_max(s_rho[tid]);
        double b = warp_max(s_b  [tid]);
        if (tid == 0) { s_rho[0] = r; s_b[0] = b; }
    }
    __syncthreads();

    if (tid == 0) {
        double rho = s_rho[0];
        int J;
        if (rho == 0.0 || rho >= 0.5) {
            J = n_row - 1;
        } else {
            double max_b = s_b[0];
            double lambda_p = (1.0 + sqrt(1.0 - 4.0 * rho * rho)) * 0.5;
            double q = rho / lambda_p;
            double B = q * (2.0 + q) / ((1.0 - q) * (1.0 - 2.0 * rho)) * max_b;
            int Jcomp = (int)(log(eps / B) / log(q)) + 1;
            J = (Jcomp < n_row - 1) ? Jcomp : n_row - 1;
        }
        *d_J = J;
    }
}

// Single-block GPU implementation of cal_J_v2 (D0, DN passed as host scalars).
__global__ void k_cal_J_v2(int* __restrict__ d_J,
                           const double* __restrict__ d_A_rho,
                           const double* __restrict__ d_C_rho,
                           double D0, double DN,
                           int n_row, double eps) {
    __shared__ double s_rho[BLOCK_SYS];
    int tid = threadIdx.x;

    double my_rho = 0.0;
    for (int k = 2 + tid; k < n_row - 1; k += blockDim.x)
        my_rho = fmax(my_rho, fmax(fabs(d_A_rho[k]), fabs(d_C_rho[k])));

    s_rho[tid] = my_rho;
    __syncthreads();
    for (int s = BLOCK_SYS / 2; s >= 32; s >>= 1) {
        if (tid < s) s_rho[tid] = fmax(s_rho[tid], s_rho[tid + s]);
        __syncthreads();
    }
    if (tid < 32) {
        double r = warp_max(s_rho[tid]);
        if (tid == 0) s_rho[0] = r;
    }
    __syncthreads();

    if (tid == 0) {
        double rho = s_rho[0];
        int J;
        if (rho == 0.0 || rho >= 0.5) {
            J = n_row - 1;
        } else {
            double abs_D0 = fabs(D0);
            double abs_DN = fabs(DN);
            double lambda_p = (1.0 + sqrt(1.0 - 4.0 * rho * rho)) * 0.5;
            double q = rho / lambda_p;
            double B = (abs_D0 + abs_DN) * q / (1.0 - q) + abs_D0 * q * q / (1.0 - q);
            int Jcomp = (int)(log(eps / B) / log(q)) + 1;
            J = (Jcomp < n_row - 1) ? Jcomp : n_row - 1;
        }
        *d_J = J;
    }
}

// Pack the j=(n_row-1) row of a [n_row × n_sys] device array into a contiguous
// n_sys-double buffer. Used to stage send data through dedicated MPI buffers
// so the CUDA-aware MPI fast path (plain MPI_DOUBLE on contiguous device
// pointer) is taken — mirrors PaScaL_TDMA's pack_rd2send_kernel pattern.
__global__ void k_pack_lastrow(double* __restrict__ dst,
                               const double* __restrict__ src,
                               int n_sys, int n_row) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;
    dst[i] = src[(std::size_t)(n_row - 1) * n_sys + i];
}

// Single bidirectional exchange: solve BOTH interface 2x2 blocks locally
// (one comm round, replacing the old forward-solve + back-communication).
//   LEFT  : D0 = (D0 - A0*D_left)   / (1 - C_left*A0)
//   RIGHT : DN = (DN - CN*D0_right) / (1 - A_right*CN)
__global__ void k_solve_both(double* __restrict__ D,
                             const double* __restrict__ A,
                             const double* __restrict__ C,
                             const double* __restrict__ C_left,
                             const double* __restrict__ D_left,
                             const double* __restrict__ A_right,
                             const double* __restrict__ D0_right,
                             int has_left, int has_right, int n_sys, int n_row) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;
    if (has_left) {
        double A0 = A[i], D0 = D[i];
        D[i] = (D0 - A0 * D_left[i]) / (1.0 - C_left[i] * A0);
    }
    if (has_right) {
        std::size_t off = (std::size_t)(n_row - 1) * n_sys + i;
        double CN = C[off], DN = D[off];
        D[off] = (DN - CN * D0_right[i]) / (1.0 - A_right[i] * CN);
    }
}

// =============================================================================
//  Constructor / destructor
// =============================================================================

FilteredTDMACUDA::FilteredTDMACUDA(int n_sys, int n_row,
                                   int myrank, int nprocs, MPI_Comm comm,
                                   int left_rank, int right_rank,
                                   double eps_constant)
    : comm_(comm), n_sys_(n_sys), n_row_(n_row), nprocs_(nprocs),
      left_rank_(left_rank), right_rank_(right_rank)
{
    (void)myrank;

    // Dedicated contiguous device buffers for the CUDA-aware MPI fast path.
    CUDA_CHECK(cudaMalloc(&d_C_lastrow_send_, sizeof(double) * n_sys));
    CUDA_CHECK(cudaMalloc(&d_D_lastrow_send_, sizeof(double) * n_sys));
    CUDA_CHECK(cudaMalloc(&d_C_left_recv_,    sizeof(double) * n_sys));
    CUDA_CHECK(cudaMalloc(&d_D_left_recv_,    sizeof(double) * n_sys));
    CUDA_CHECK(cudaMalloc(&d_A_right_recv_,   sizeof(double) * n_sys));
    CUDA_CHECK(cudaMalloc(&d_D0_right_recv_,  sizeof(double) * n_sys));

    // Per-row spectral-radius estimates + scalar J output (used by cal_J).
    CUDA_CHECK(cudaMalloc(&d_A_rho_, sizeof(double) * (n_row + 1)));
    CUDA_CHECK(cudaMalloc(&d_C_rho_, sizeof(double) * (n_row + 1)));
    CUDA_CHECK(cudaMemset(d_A_rho_, 0, sizeof(double) * (n_row + 1)));
    CUDA_CHECK(cudaMemset(d_C_rho_, 0, sizeof(double) * (n_row + 1)));
    CUDA_CHECK(cudaMalloc(&d_J_, sizeof(int)));

    int N = n_row * nprocs;
    eps_ = eps_constant / ((double)N * N);
}

FilteredTDMACUDA::~FilteredTDMACUDA() {
    if (d_C_lastrow_send_) cudaFree(d_C_lastrow_send_);
    if (d_D_lastrow_send_) cudaFree(d_D_lastrow_send_);
    if (d_C_left_recv_)    cudaFree(d_C_left_recv_);
    if (d_D_left_recv_)    cudaFree(d_D_left_recv_);
    if (d_A_right_recv_)   cudaFree(d_A_right_recv_);
    if (d_D0_right_recv_)  cudaFree(d_D0_right_recv_);
    if (d_A_rho_)          cudaFree(d_A_rho_);
    if (d_C_rho_)          cudaFree(d_C_rho_);
    if (d_J_)              cudaFree(d_J_);
    if (d_E_)              cudaFree(d_E_);
}

// =============================================================================
//  set_rho_device — fill d_A_rho_/d_C_rho_ from current matrix entries
// =============================================================================

void FilteredTDMACUDA::set_rho_device(const double* d_A,
                                      const double* d_B,
                                      const double* d_C) {
    const int n = n_row_ + 1;
    dim3 block(BLOCK_SYS), grid((n + BLOCK_SYS - 1) / BLOCK_SYS);
    k_set_rho<<<grid, block>>>(d_A_rho_, d_C_rho_, d_A, d_B, d_C,
                               n_sys_, n_row_);
}

// =============================================================================
//  Cutoff-J estimators (host-side, with small device→host copies)
// =============================================================================

int FilteredTDMACUDA::cal_J_v1(const double* D0_dev, const double* DN_dev) {
    // Launch single-block reduction kernel: computes rho + max|D0| + max|DN| +
    // J on the device, writes one int back. Host does only the cudaMemcpy of
    // the result — no host-side libm calls (avoids the NVHPC nvc++ SIGILL).
    k_cal_J_v1<<<1, BLOCK_SYS>>>(d_J_, d_A_rho_, d_C_rho_,
                                 D0_dev, DN_dev, n_sys_, n_row_, eps_);
    int J = 0;
    CUDA_CHECK(cudaMemcpy(&J, d_J_, sizeof(int), cudaMemcpyDeviceToHost));
    return J;
}

int FilteredTDMACUDA::cal_J_v2(double D0, double DN) {
    k_cal_J_v2<<<1, BLOCK_SYS>>>(d_J_, d_A_rho_, d_C_rho_,
                                 D0, DN, n_row_, eps_);
    int J = 0;
    CUDA_CHECK(cudaMemcpy(&J, d_J_, sizeof(int), cudaMemcpyDeviceToHost));
    return J;
}

int FilteredTDMACUDA::cal_J_rhs_bound(const double* d_D) {
    k_cal_J_rhs_bound<<<1, BLOCK_SYS>>>(d_J_, d_A_rho_, d_C_rho_,
                                        d_D, n_sys_, n_row_, eps_);
    int J = 0;
    CUDA_CHECK(cudaMemcpy(&J, d_J_, sizeof(int), cudaMemcpyDeviceToHost));
    return J;
}

// =============================================================================
//  solve_filtered_v1
// =============================================================================

void FilteredTDMACUDA::solve_filtered_v1(double* d_A, double* d_B, double* d_C, double* d_D) {
    if (nprocs_ == 1) { tdma_many_cuda(d_A, d_B, d_C, d_D, n_sys_, n_row_); return; }
    v1_multirank(d_A, d_B, d_C, d_D);
}

void FilteredTDMACUDA::solve_cycl_filtered_v1(double* d_A, double* d_B, double* d_C, double* d_D) {
    // Periodic comm makes both interfaces (incl. the wrap-around) real; the
    // multi-rank body is identical to the non-cyclic solve.
    if (nprocs_ == 1) {
        if (!d_E_) CUDA_CHECK(cudaMalloc(&d_E_, sizeof(double) * (std::size_t)n_sys_ * n_row_));
        tdma_cyclic_many_cuda(d_A, d_B, d_C, d_D, d_E_, n_sys_, n_row_);
        return;
    }
    v1_multirank(d_A, d_B, d_C, d_D);
}

void FilteredTDMACUDA::v1_multirank(double* d_A, double* d_B, double* d_C, double* d_D) {
    const dim3 block(BLOCK_SYS);
    const dim3 grid = grid1D(n_sys_);

    // 1) Forward elimination (rows 0,1 normalize + 2..n_row-1) — single launch.
    k_fwd_pass<<<grid, block>>>(d_A, d_B, d_C, d_D, n_sys_, n_row_);

    // 2) Backward substitution (rows n_row-3..1) — single launch.
    k_bwd_pass<<<grid, block>>>(d_A, d_C, d_D, n_sys_, n_row_);

    // 3) Pack reduced system
    k_pack<<<grid, block>>>(d_A, d_C, d_D, n_sys_);

    // 4) Single bidirectional exchange (merged): row N-1 (C,D) -> right,
    //    row 0 (A,D) -> left. The reduced system decouples into independent
    //    interface 2x2 blocks, so each rank solves BOTH interfaces locally in
    //    one comm round (was: forward exchange + solve + back-communication).
    //    Row 0 (d_A,d_D first n_sys) is already contiguous → sent directly.
    k_pack_lastrow<<<grid, block>>>(d_C_lastrow_send_, d_C, n_sys_, n_row_);
    k_pack_lastrow<<<grid, block>>>(d_D_lastrow_send_, d_D, n_sys_, n_row_);
    CUDA_CHECK(cudaStreamSynchronize(0));
    {
        MPI_Request req[8];
        int nreq = 0;
        MPI_Isend(d_C_lastrow_send_, n_sys_, MPI_DOUBLE, right_rank_, 1, comm_, &req[nreq++]);
        MPI_Irecv(d_C_left_recv_,    n_sys_, MPI_DOUBLE, left_rank_,  1, comm_, &req[nreq++]);
        MPI_Isend(d_D_lastrow_send_, n_sys_, MPI_DOUBLE, right_rank_, 2, comm_, &req[nreq++]);
        MPI_Irecv(d_D_left_recv_,    n_sys_, MPI_DOUBLE, left_rank_,  2, comm_, &req[nreq++]);
        MPI_Isend(d_A,               n_sys_, MPI_DOUBLE, left_rank_,  3, comm_, &req[nreq++]);
        MPI_Irecv(d_A_right_recv_,   n_sys_, MPI_DOUBLE, right_rank_, 3, comm_, &req[nreq++]);
        MPI_Isend(d_D,               n_sys_, MPI_DOUBLE, left_rank_,  4, comm_, &req[nreq++]);
        MPI_Irecv(d_D0_right_recv_,  n_sys_, MPI_DOUBLE, right_rank_, 4, comm_, &req[nreq++]);
        MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
    }

    // 5) Solve both interface 2x2 blocks locally (no second round).
    k_solve_both<<<grid, block>>>(d_D, d_A, d_C,
                                  d_C_left_recv_, d_D_left_recv_,
                                  d_A_right_recv_, d_D0_right_recv_,
                                  (left_rank_  != MPI_PROC_NULL) ? 1 : 0,
                                  (right_rank_ != MPI_PROC_NULL) ? 1 : 0,
                                  n_sys_, n_row_);

    // 8) Compute J on device (no D2H), then apply final corrections.
    const double* D0_dev = d_D;
    const double* DN_dev = d_D + (std::size_t)(n_row_ - 1) * n_sys_;
    k_cal_J_v1<<<1, BLOCK_SYS>>>(d_J_, d_A_rho_, d_C_rho_,
                                  D0_dev, DN_dev, n_sys_, n_row_, eps_);
    k_final_pass<<<grid, block>>>(d_A, d_C, d_D, D0_dev, DN_dev,
                                  n_sys_, n_row_, d_J_);
}

// =============================================================================
//  solve_filtered_v2
// =============================================================================

void FilteredTDMACUDA::solve_filtered_v2(double* d_A, double* d_B, double* d_C, double* d_D) {
    if (nprocs_ == 1) { tdma_many_cuda(d_A, d_B, d_C, d_D, n_sys_, n_row_); return; }
    v2_multirank(d_A, d_B, d_C, d_D);
}

void FilteredTDMACUDA::solve_cycl_filtered_v2(double* d_A, double* d_B, double* d_C, double* d_D) {
    if (nprocs_ == 1) {
        if (!d_E_) CUDA_CHECK(cudaMalloc(&d_E_, sizeof(double) * (std::size_t)n_sys_ * n_row_));
        tdma_cyclic_many_cuda(d_A, d_B, d_C, d_D, d_E_, n_sys_, n_row_);
        return;
    }
    v2_multirank(d_A, d_B, d_C, d_D);
}

void FilteredTDMACUDA::v2_multirank(double* d_A, double* d_B, double* d_C, double* d_D) {
    const dim3 block(BLOCK_SYS);
    const dim3 grid = grid1D(n_sys_);

    // Compute J from boundary rows only (n_sys elements each, not n_row*n_sys).
    // k_cal_J_v1 reads D0 and DN — the pre-solve first/last rows of d_D — which
    // is n_row× cheaper than the full-RHS scan of k_cal_J_rhs_bound.
    const double* D0_pre = d_D;
    const double* DN_pre = d_D + (std::size_t)(n_row_ - 1) * n_sys_;
    k_cal_J_v1<<<1, BLOCK_SYS>>>(d_J_, d_A_rho_, d_C_rho_,
                                  D0_pre, DN_pre, n_sys_, n_row_, eps_);

    // 1) Forward elimination — reads J from d_J_ via __ldg.
    k_fwd_pass_v2<<<grid, block>>>(d_A, d_B, d_C, d_D, n_sys_, n_row_, d_J_);

    // 2) Backward substitution — reads J (and derives lo) from d_J_.
    k_bwd_pass_v2<<<grid, block>>>(d_A, d_C, d_D, n_sys_, n_row_, d_J_);

    // 3) Pack reduced system
    k_pack<<<grid, block>>>(d_A, d_C, d_D, n_sys_);

    // 4) Single bidirectional exchange (merged — see solve_filtered_v1):
    //    row N-1 (C,D) -> right ; row 0 (A,D, contiguous) -> left. One round.
    k_pack_lastrow<<<grid, block>>>(d_C_lastrow_send_, d_C, n_sys_, n_row_);
    k_pack_lastrow<<<grid, block>>>(d_D_lastrow_send_, d_D, n_sys_, n_row_);
    CUDA_CHECK(cudaStreamSynchronize(0));
    {
        MPI_Request req[8];
        int nreq = 0;
        MPI_Isend(d_C_lastrow_send_, n_sys_, MPI_DOUBLE, right_rank_, 1, comm_, &req[nreq++]);
        MPI_Irecv(d_C_left_recv_,    n_sys_, MPI_DOUBLE, left_rank_,  1, comm_, &req[nreq++]);
        MPI_Isend(d_D_lastrow_send_, n_sys_, MPI_DOUBLE, right_rank_, 2, comm_, &req[nreq++]);
        MPI_Irecv(d_D_left_recv_,    n_sys_, MPI_DOUBLE, left_rank_,  2, comm_, &req[nreq++]);
        MPI_Isend(d_A,               n_sys_, MPI_DOUBLE, left_rank_,  3, comm_, &req[nreq++]);
        MPI_Irecv(d_A_right_recv_,   n_sys_, MPI_DOUBLE, right_rank_, 3, comm_, &req[nreq++]);
        MPI_Isend(d_D,               n_sys_, MPI_DOUBLE, left_rank_,  4, comm_, &req[nreq++]);
        MPI_Irecv(d_D0_right_recv_,  n_sys_, MPI_DOUBLE, right_rank_, 4, comm_, &req[nreq++]);
        MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
    }

    // 5) Solve both interface 2x2 blocks locally (no second round).
    k_solve_both<<<grid, block>>>(d_D, d_A, d_C,
                                  d_C_left_recv_, d_D_left_recv_,
                                  d_A_right_recv_, d_D0_right_recv_,
                                  (left_rank_  != MPI_PROC_NULL) ? 1 : 0,
                                  (right_rank_ != MPI_PROC_NULL) ? 1 : 0,
                                  n_sys_, n_row_);

    // 8) Final corrections — d_J_ already holds the correct J from step 1.
    const double* D0_dev = d_D;
    const double* DN_dev = d_D + (std::size_t)(n_row_ - 1) * n_sys_;
    k_final_pass<<<grid, block>>>(d_A, d_C, d_D, D0_dev, DN_dev,
                                  n_sys_, n_row_, d_J_);
    (void)d_B;
}
