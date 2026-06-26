#include "filtered_tdma_cuda.hpp"
#include "tdma_local_cuda.cuh"

#include <cuda_runtime.h>
#include <cstddef>
#include <cmath>
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

// Shared-memory + warp-shuffle tree reduction for one double array.
// After the call s_a[0] holds the maximum; all threads see the result.
__device__ inline void reduce1_max(double* s_a, int tid) {
    for (int s = BLOCK_SYS / 2; s >= 32; s >>= 1) {
        if (tid < s) s_a[tid] = fmax(s_a[tid], s_a[tid + s]);
        __syncthreads();
    }
    if (tid < 32) {
        double a = warp_max(s_a[tid]);
        if (tid == 0) s_a[0] = a;
    }
    __syncthreads();
}

// Same as reduce1_max but reduces three arrays simultaneously in one pass.
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

// v1 forward elimination. Normalizes rows 0 and 1 first, then eliminates
// rows 2..n_row-1 with a register-resident carry (a_prev/c_prev/d_prev).
__global__ __launch_bounds__(BLOCK_SYS)
void k_fwd_pass_v1(double* A, double* B, double* C, double* D,
                int n_sys, int n_row) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;

    double r0 = 1.0 / B[i];
    A[i] *= r0;  C[i] *= r0;  D[i] *= r0;

    std::size_t off1 = (std::size_t)1 * n_sys + i;
    double r1 = 1.0 / B[off1];
    A[off1] *= r1;  C[off1] *= r1;  D[off1] *= r1;

    double a_prev = A[off1], c_prev = C[off1], d_prev = D[off1];

    for (int j = 2; j < n_row; ++j) {
        std::size_t off_j = (std::size_t)j * n_sys + i;
        double Aj = A[off_j], Bj = B[off_j], Cj = C[off_j], Dj = D[off_j];
        double inv  = 1.0 / (Bj - Aj * c_prev);
        double d_new =  inv * (Dj - Aj * d_prev);
        double c_new =  inv * Cj;
        double a_new = -inv * Aj * a_prev;
        D[off_j] = d_new;  C[off_j] = c_new;  A[off_j] = a_new;
        a_prev = a_new;  c_prev = c_new;  d_prev = d_new;
    }
}

// v1 backward substitution + row-0 pack in one kernel.
// After the backward loop (rows n_row-3..1), a_next/d_next hold A[1]/D[1]
// in registers; row-0 decoupling is appended without a separate launch.
__global__ __launch_bounds__(BLOCK_SYS)
void k_bwd_pass_v1(double* A, double* C, double* D,
                int n_sys, int n_row) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;

    std::size_t off_nm2 = (std::size_t)(n_row - 2) * n_sys + i;
    double d_next = D[off_nm2], a_next = A[off_nm2], c_next = C[off_nm2];

    for (int j = n_row - 3; j >= 1; --j) {
        std::size_t off_j = (std::size_t)j * n_sys + i;
        double Cj   = C[off_j];
        double d_new = D[off_j] - Cj * d_next;
        double a_new = A[off_j] - Cj * a_next;
        double c_new = -Cj * c_next;
        D[off_j] = d_new;  A[off_j] = a_new;  C[off_j] = c_new;
        d_next = d_new;  a_next = a_new;  c_next = c_new;
    }
    // Pack row 0: a_next == A[1], d_next == D[1] already in registers.
    double C0 = C[i], D0_val = D[i];
    double r  = 1.0 / (1.0 - a_next * C0);
    D[i] = r * (D0_val - C0 * d_next);
    A[i] = r * A[i];
}

// v2 forward elimination. Two loops split at cutoff J:
//   j in [2, J]       : full update (A, C, D) with register carry.
//   j in [J+1, n_row-1]: skip-A update (C, D only) with register carry.
__global__ __launch_bounds__(BLOCK_SYS)
void k_fwd_pass_v2(double* A, double* B, double* C, double* D,
                   int n_sys, int n_row, const int* __restrict__ d_J) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;
    const int J = __ldg(d_J);

    double r0 = 1.0 / B[i];
    A[i] *= r0;  C[i] *= r0;  D[i] *= r0;

    std::size_t off1 = (std::size_t)1 * n_sys + i;
    double r1 = 1.0 / B[off1];
    A[off1] *= r1;  C[off1] *= r1;  D[off1] *= r1;

    double a_prev = A[off1], c_prev = C[off1], d_prev = D[off1];

    for (int j = 2; j <= J; ++j) {
        std::size_t off_j = (std::size_t)j * n_sys + i;
        double Aj = A[off_j], Bj = B[off_j], Cj = C[off_j], Dj = D[off_j];
        double inv  = 1.0 / (Bj - Aj * c_prev);
        double d_new =  inv * (Dj - Aj * d_prev);
        double c_new =  inv * Cj;
        double a_new = -inv * Aj * a_prev;
        D[off_j] = d_new;  C[off_j] = c_new;  A[off_j] = a_new;
        a_prev = a_new;  c_prev = c_new;  d_prev = d_new;
    }
    // skip-A loop: A[j] stays unchanged for j > J; carry c_prev, d_prev only.
    for (int j = J + 1; j < n_row; ++j) {
        std::size_t off_j = (std::size_t)j * n_sys + i;
        double Aj = A[off_j], Bj = B[off_j], Cj = C[off_j], Dj = D[off_j];
        double inv  = 1.0 / (Bj - Aj * c_prev);
        double d_new = inv * (Dj - Aj * d_prev);
        double c_new = inv * Cj;
        D[off_j] = d_new;  C[off_j] = c_new;
        c_prev = c_new;  d_prev = d_new;
    }
}

// v2 backward substitution + row-0 pack in one kernel. Single merged pass:
//   For j = n_row-3 down to 1, read C[j] once, then:
//     - D back-substitute (all rows):  D[j] -= C[j]*d_next
//     - A back-substitute (j < J):     A[j] -= C[j]*a_next
//     - C multiply       (j > lo):     C[j]  = -C[j]*c_next  (written last)
//   Row-0 decoupling appended after the loop.
// lo = (n_row-2) - J.  C[j] is read once vs. 3x in the original 3-pass version.
__global__ __launch_bounds__(BLOCK_SYS)
void k_bwd_pass_v2(double* A, double* C, double* D,
                   int n_sys, int n_row, const int* __restrict__ d_J) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;
    const int J  = __ldg(d_J);
    const int lo = (n_row - 2) - J;

    std::size_t off_nm2 = (std::size_t)(n_row - 2) * n_sys + i;
    double d_next = D[off_nm2];
    double c_next = C[off_nm2];
    double a_next = (J >= 2) ? A[(std::size_t)J * n_sys + i] : 0.0;

    for (int j = n_row - 3; j >= 1; --j) {
        std::size_t off_j = (std::size_t)j * n_sys + i;
        double Cj = C[off_j];   // read original C[j] once before any write

        // D back-substitute (all rows)
        double d_new = D[off_j] - Cj * d_next;
        D[off_j] = d_new;
        d_next = d_new;

        // A back-substitute (rows j < J)
        if (J >= 2 && j < J) {
            double a_new = A[off_j] - Cj * a_next;
            A[off_j] = a_new;
            a_next = a_new;
        }

        // C multiply (rows j > lo) — overwrites C[j] after Cj already consumed
        if (j > lo) {
            double c_new = -Cj * c_next;
            C[off_j] = c_new;
            c_next = c_new;
        }
    }

    // Pack row 0: A[1] and D[1] updated by the loop above.
    double A1    = A[(std::size_t)1 * n_sys + i];
    double C0    = C[i];
    double D0_val = D[i];
    double D1    = D[(std::size_t)1 * n_sys + i];
    double r     = 1.0 / (1.0 - A1 * C0);
    D[i] = r * (D0_val - C0 * D1);
    A[i] = r * A[i];
}

// Final correction: applies interface corrections using boundary values D0/DN.
//   Left  correction: D[j] -= A[j]*D0  for j in [1, J].
//   Right correction: D[j] -= C[j]*DN  for j in [n_row-1-J, n_row-2].
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


// Compute J from a scalar RHS bound. Takes max_b = |D| directly (no array scan).
// Useful when a single representative RHS magnitude is already known.
__global__ void k_cal_J_v2(int* __restrict__ d_J,
                            const double* __restrict__ d_A_rho,
                            const double* __restrict__ d_C_rho,
                            double D,
                            int n_row, double eps) {
    __shared__ double s_rho[BLOCK_SYS];
    int tid = threadIdx.x;

    double my_rho = 0.0;
    for (int k = 2 + tid; k < n_row - 1; k += blockDim.x)
        my_rho = fmax(my_rho, fmax(fabs(d_A_rho[k]), fabs(d_C_rho[k])));

    s_rho[tid] = my_rho;
    __syncthreads();
    reduce1_max(s_rho, tid);

    if (tid == 0) {
        double rho   = s_rho[0];
        double max_b = fabs(D);
        int J;
        if (rho == 0.0 || rho >= 0.5) {
            J = n_row - 1;
        } else {
            double lambda_p = (1.0 + sqrt(1.0 - 4.0 * rho * rho)) * 0.5;
            double q = rho / lambda_p;
            double B = q * (2.0 + q) / ((1.0 - q) * (1.0 - 2.0 * rho)) * max_b;
            int Jcomp = (int)(log(eps / B) / log(q)) + 1;
            J = (Jcomp < n_row - 1) ? Jcomp : n_row - 1;
            if (J < 0) J = 0;
        }
        *d_J = J;
    }
}

// Compute J from boundary row RHS values. Reduces max(|D0[i]|, |DN[i]|) over
// n_sys elements and max rho over d_A_rho/d_C_rho, then applies the v2 formula:
//   B = q(2+q) / ((1-q)(1-2*rho)) * max_b,  J = floor(log(eps/B) / log(q)) + 1
__global__ void k_cal_J_rhs_bound(int* __restrict__ d_J,
                                   const double* __restrict__ d_A_rho,
                                   const double* __restrict__ d_C_rho,
                                   const double* __restrict__ D0,
                                   const double* __restrict__ DN,
                                   int n_sys, int n_row, double eps) {
    __shared__ double s_rho[BLOCK_SYS];
    __shared__ double s_D0 [BLOCK_SYS];
    __shared__ double s_DN [BLOCK_SYS];
    int tid = threadIdx.x;

    double my_rho = 0.0;
    for (int k = 2 + tid; k < n_row - 1; k += blockDim.x)
        my_rho = fmax(my_rho, fmax(fabs(d_A_rho[k]), fabs(d_C_rho[k])));

    double my_D0 = 0.0, my_DN = 0.0;
    for (int i = tid; i < n_sys; i += blockDim.x) {
        my_D0 = fmax(my_D0, fabs(D0[i]));
        my_DN = fmax(my_DN, fabs(DN[i]));
    }

    s_rho[tid] = my_rho; s_D0[tid] = my_D0; s_DN[tid] = my_DN;
    __syncthreads();
    reduce3_max(s_rho, s_D0, s_DN, tid);

    if (tid == 0) {
        double rho   = s_rho[0];
        double max_b = fmax(s_D0[0], s_DN[0]);
        int J;
        if (rho == 0.0 || rho >= 0.5) {
            J = n_row - 1;
        } else {
            double lambda_p = (1.0 + sqrt(1.0 - 4.0 * rho * rho)) * 0.5;
            double q = rho / lambda_p;
            double B = q * (2.0 + q) / ((1.0 - q) * (1.0 - 2.0 * rho)) * max_b;
            int Jcomp = (int)(log(eps / B) / log(q)) + 1;
            J = (Jcomp < n_row - 1) ? Jcomp : n_row - 1;
            if (J < 0) J = 0;
        }
        *d_J = J;
    }
}

// Pack the last row into one 2*n_sys send buffer for the right neighbor:
//   dst[0..n_sys-1] = C[n_row-1, :],  dst[n_sys..2*n_sys-1] = D[n_row-1, :]
__global__ void k_pack_lastrow(double* __restrict__ dst,
                                const double* __restrict__ C,
                                const double* __restrict__ D,
                                int n_sys, int n_row) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;
    std::size_t last = (std::size_t)(n_row - 1) * n_sys;
    dst[i]         = C[last + i];
    dst[n_sys + i] = D[last + i];
}

// Pack row 0 into one 2*n_sys send buffer for the left neighbor:
//   dst[0..n_sys-1] = A[0, :],  dst[n_sys..2*n_sys-1] = D[0, :]
__global__ void k_pack_row0(double* __restrict__ dst,
                             const double* __restrict__ A,
                             const double* __restrict__ D,
                             int n_sys) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;
    dst[i]         = A[i];
    dst[n_sys + i] = D[i];
}

// Solve both interface 2×2 blocks in place after the MPI exchange:
//   left  boundary: D[0]     = (D[0] - A[0]*D_left)    / (1 - C_left*A[0])
//   right boundary: D[n-1]   = (D[n-1] - C[n-1]*D0_right) / (1 - A_right*C[n-1])
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

    // MPI exchange buffers — skip for local (nprocs==1) solvers; they never use
    // these buffers and large n_sys (e.g. ix*iz for the y-direction) would
    // exhaust per-process CUDA allocation limits on multi-GPU nodes.
    if (nprocs > 1) {
        CUDA_CHECK(cudaMalloc(&d_send_right_, sizeof(double) * 2 * n_sys));
        CUDA_CHECK(cudaMalloc(&d_send_left_,  sizeof(double) * 2 * n_sys));
        CUDA_CHECK(cudaMalloc(&d_recv_left_,  sizeof(double) * 2 * n_sys));
        CUDA_CHECK(cudaMalloc(&d_recv_right_, sizeof(double) * 2 * n_sys));
    }

    // Per-row spectral-radius estimates + scalar J output (used by cal_J).
    CUDA_CHECK(cudaMalloc(&d_A_rho_, sizeof(double) * (n_row + 1)));
    CUDA_CHECK(cudaMalloc(&d_C_rho_, sizeof(double) * (n_row + 1)));
    CUDA_CHECK(cudaMemset(d_A_rho_, 0, sizeof(double) * (n_row + 1)));
    CUDA_CHECK(cudaMemset(d_C_rho_, 0, sizeof(double) * (n_row + 1)));
    CUDA_CHECK(cudaMalloc(&d_J_, sizeof(int)));

    CUDA_CHECK(cudaEventCreate(&e_gpu_start_));
    CUDA_CHECK(cudaEventCreate(&e_gpu_pre_end_));
    CUDA_CHECK(cudaEventCreate(&e_gpu_post_start_));
    CUDA_CHECK(cudaEventCreate(&e_gpu_post_end_));

    int N = n_row * nprocs;
    eps_ = eps_constant / ((double)N * N);
}

FilteredTDMACUDA::~FilteredTDMACUDA() {
    if (d_send_right_) cudaFree(d_send_right_);
    if (d_send_left_)  cudaFree(d_send_left_);
    if (d_recv_left_)  cudaFree(d_recv_left_);
    if (d_recv_right_) cudaFree(d_recv_right_);
    if (d_A_rho_)          cudaFree(d_A_rho_);
    if (d_C_rho_)          cudaFree(d_C_rho_);
    if (d_J_)              cudaFree(d_J_);
    if (d_E_)              cudaFree(d_E_);
    if (e_gpu_start_)      cudaEventDestroy(e_gpu_start_);
    if (e_gpu_pre_end_)    cudaEventDestroy(e_gpu_pre_end_);
    if (e_gpu_post_start_) cudaEventDestroy(e_gpu_post_start_);
    if (e_gpu_post_end_)   cudaEventDestroy(e_gpu_post_end_);
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
//  Cutoff-J estimator
// =============================================================================

int FilteredTDMACUDA::cal_J_rhs_bound(const double* d_D) {
    const double* D0_dev = d_D;
    const double* DN_dev = d_D + (std::size_t)(n_row_ - 1) * n_sys_;
    k_cal_J_rhs_bound<<<1, BLOCK_SYS>>>(d_J_, d_A_rho_, d_C_rho_,
                                        D0_dev, DN_dev, n_sys_, n_row_, eps_);
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

    cudaEventRecord(e_gpu_start_);

    // 1) Forward elimination.
    k_fwd_pass_v1<<<grid, block>>>(d_A, d_B, d_C, d_D, n_sys_, n_row_);
    // 2) Backward substitution + row-0 pack.
    k_bwd_pass_v1<<<grid, block>>>(d_A, d_C, d_D, n_sys_, n_row_);
    // 3) Pack boundary rows into send buffers.
    k_pack_lastrow<<<grid, block>>>(d_send_right_, d_C, d_D, n_sys_, n_row_);
    k_pack_row0<<<grid, block>>>(d_send_left_, d_A, d_D, n_sys_);
    cudaEventRecord(e_gpu_pre_end_);

    // 4) MPI exchange (measured after GPU sync).
    CUDA_CHECK(cudaStreamSynchronize(0));
    {
        double t0 = MPI_Wtime();
        MPI_Request req[4];
        int nreq = 0;
        MPI_Isend(d_send_right_, 2 * n_sys_, MPI_DOUBLE, right_rank_, 1, comm_, &req[nreq++]);
        MPI_Irecv(d_recv_left_,  2 * n_sys_, MPI_DOUBLE, left_rank_,  1, comm_, &req[nreq++]);
        MPI_Isend(d_send_left_,  2 * n_sys_, MPI_DOUBLE, left_rank_,  2, comm_, &req[nreq++]);
        MPI_Irecv(d_recv_right_, 2 * n_sys_, MPI_DOUBLE, right_rank_, 2, comm_, &req[nreq++]);
        MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
        last_comm_ms_ = (MPI_Wtime() - t0) * 1e3;
    }

    cudaEventRecord(e_gpu_post_start_);
    // 5) Solve both interface 2×2 blocks.
    k_solve_both<<<grid, block>>>(d_D, d_A, d_C,
                                  d_recv_left_,           // C from left
                                  d_recv_left_  + n_sys_, // D from left
                                  d_recv_right_,          // A from right
                                  d_recv_right_ + n_sys_, // D0 from right
                                  (left_rank_  != MPI_PROC_NULL) ? 1 : 0,
                                  (right_rank_ != MPI_PROC_NULL) ? 1 : 0,
                                  n_sys_, n_row_);
    // 6) Compute J, then apply final corrections.
    const double* D0_dev = d_D;
    const double* DN_dev = d_D + (std::size_t)(n_row_ - 1) * n_sys_;
    k_cal_J_rhs_bound<<<1, BLOCK_SYS>>>(d_J_, d_A_rho_, d_C_rho_,
                                        D0_dev, DN_dev, n_sys_, n_row_, eps_);
    k_final_pass<<<grid, block>>>(d_A, d_C, d_D, D0_dev, DN_dev,
                                  n_sys_, n_row_, d_J_);
    cudaEventRecord(e_gpu_post_end_);

    cudaEventSynchronize(e_gpu_post_end_);
    float ms_pre = 0.0f, ms_post = 0.0f;
    cudaEventElapsedTime(&ms_pre,  e_gpu_start_,      e_gpu_pre_end_);
    cudaEventElapsedTime(&ms_post, e_gpu_post_start_,  e_gpu_post_end_);
    last_gpu_ms_ = ms_pre + ms_post;
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

    cudaEventRecord(e_gpu_start_);

    // 1) Compute J from boundary rows (before forward pass modifies d_D).
    {
        const double* D0_pre = d_D;
        const double* DN_pre = d_D + (std::size_t)(n_row_ - 1) * n_sys_;
        k_cal_J_rhs_bound<<<1, BLOCK_SYS>>>(d_J_, d_A_rho_, d_C_rho_,
                                            D0_pre, DN_pre, n_sys_, n_row_, eps_);
    }
    // 2) Forward elimination (uses J from step 1).
    k_fwd_pass_v2<<<grid, block>>>(d_A, d_B, d_C, d_D, n_sys_, n_row_, d_J_);
    // 3) Backward substitution + row-0 pack.
    k_bwd_pass_v2<<<grid, block>>>(d_A, d_C, d_D, n_sys_, n_row_, d_J_);
    // 4) Pack boundary rows into send buffers.
    k_pack_lastrow<<<grid, block>>>(d_send_right_, d_C, d_D, n_sys_, n_row_);
    k_pack_row0<<<grid, block>>>(d_send_left_, d_A, d_D, n_sys_);
    cudaEventRecord(e_gpu_pre_end_);

    // 5) MPI exchange (measured after GPU sync).
    CUDA_CHECK(cudaStreamSynchronize(0));
    {
        double t0 = MPI_Wtime();
        MPI_Request req[4];
        int nreq = 0;
        MPI_Isend(d_send_right_, 2 * n_sys_, MPI_DOUBLE, right_rank_, 1, comm_, &req[nreq++]);
        MPI_Irecv(d_recv_left_,  2 * n_sys_, MPI_DOUBLE, left_rank_,  1, comm_, &req[nreq++]);
        MPI_Isend(d_send_left_,  2 * n_sys_, MPI_DOUBLE, left_rank_,  2, comm_, &req[nreq++]);
        MPI_Irecv(d_recv_right_, 2 * n_sys_, MPI_DOUBLE, right_rank_, 2, comm_, &req[nreq++]);
        MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
        last_comm_ms_ = (MPI_Wtime() - t0) * 1e3;
    }

    cudaEventRecord(e_gpu_post_start_);
    // 6) Solve both interface 2×2 blocks.
    k_solve_both<<<grid, block>>>(d_D, d_A, d_C,
                                  d_recv_left_,           // C from left
                                  d_recv_left_  + n_sys_, // D from left
                                  d_recv_right_,          // A from right
                                  d_recv_right_ + n_sys_, // D0 from right
                                  (left_rank_  != MPI_PROC_NULL) ? 1 : 0,
                                  (right_rank_ != MPI_PROC_NULL) ? 1 : 0,
                                  n_sys_, n_row_);
    // 7) Apply final corrections.
    const double* D0_dev = d_D;
    const double* DN_dev = d_D + (std::size_t)(n_row_ - 1) * n_sys_;
    k_final_pass<<<grid, block>>>(d_A, d_C, d_D, D0_dev, DN_dev,
                                  n_sys_, n_row_, d_J_);
    cudaEventRecord(e_gpu_post_end_);

    cudaEventSynchronize(e_gpu_post_end_);
    float ms_pre = 0.0f, ms_post = 0.0f;
    cudaEventElapsedTime(&ms_pre,  e_gpu_start_,       e_gpu_pre_end_);
    cudaEventElapsedTime(&ms_post, e_gpu_post_start_,   e_gpu_post_end_);
    last_gpu_ms_ = ms_pre + ms_post;
    (void)d_B;
}
