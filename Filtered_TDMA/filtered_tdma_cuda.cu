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
        double bk = d_B[(std::size_t)k * n_sys];
        if (bk != 0.0) {
            d_A_rho[k] = fabs(d_A[(std::size_t)k * n_sys] / bk);
            d_C_rho[k] = fabs(d_C[(std::size_t)k * n_sys] / bk);
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
__global__ void k_fwd_pass(double* A, double* B, double* C, double* D,
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
__global__ void k_bwd_pass(double* A, double* C, double* D,
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
__global__ void k_fwd_pass_v2(double* A, double* B, double* C, double* D,
                              int n_sys, int n_row, int J) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;

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
__global__ void k_bwd_pass_v2(double* A, double* C, double* D,
                              int n_sys, int n_row, int J, int lo) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;

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

// Merged final corrections. Applies the left correction for j in [1, j_left_end]
// and the right correction for j in [j_right_beg, n_row-2] — single launch.
__global__ void k_final_pass(const double* A, const double* C, double* D,
                             const double* D0, const double* DN,
                             int n_sys, int n_row,
                             int j_left_end, int j_right_beg) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;

    double d0 = D0[i];
    double dn = DN[i];
    int jL = j_left_end;
    if (jL > n_row - 2) jL = n_row - 2;
    int jR = j_right_beg < 1 ? 1 : j_right_beg;

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

// 5) Solve D0 from left boundary; compute left neighbor's DN.
__global__ void k_solve_D0_left(double* D, double* d_D_right_send,
                                const double* d_C_left, const double* d_D_left,
                                const double* A, int n_sys) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;
    double D0 = D[i];
    double A0 = A[i];
    double Cl = d_C_left[i];
    double Dl = d_D_left[i];
    double D0_new = (D0 - A0 * Dl) / (1.0 - Cl * A0);
    D[i] = D0_new;
    d_D_right_send[i] = Dl - Cl * D0_new;
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
    const int rho_begin = 2;
    const int rho_end   = n_row + 1 - 2;
    for (int k = rho_begin + tid; k < rho_end; k += blockDim.x) {
        double a = d_A_rho[k]; if (a < 0.0) a = -a;
        double c = d_C_rho[k]; if (c < 0.0) c = -c;
        double v = a > c ? a : c;
        if (v > my_rho) my_rho = v;
    }
    // |D0|, |DN| reductions over n_sys.
    double my_D0 = 0.0, my_DN = 0.0;
    for (int i = tid; i < n_sys; i += blockDim.x) {
        double a = D0[i]; if (a < 0.0) a = -a;
        if (a > my_D0) my_D0 = a;
        double b = DN[i]; if (b < 0.0) b = -b;
        if (b > my_DN) my_DN = b;
    }

    s_rho[tid] = my_rho;
    s_D0 [tid] = my_D0;
    s_DN [tid] = my_DN;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_rho[tid + s] > s_rho[tid]) s_rho[tid] = s_rho[tid + s];
            if (s_D0 [tid + s] > s_D0 [tid]) s_D0 [tid] = s_D0 [tid + s];
            if (s_DN [tid + s] > s_DN [tid]) s_DN [tid] = s_DN [tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        double rho = s_rho[0];
        double max_D0 = s_D0[0];
        double max_DN = s_DN[0];
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

// Single-block GPU implementation of cal_J_v2 (D0, DN passed as host scalars).
__global__ void k_cal_J_v2(int* __restrict__ d_J,
                           const double* __restrict__ d_A_rho,
                           const double* __restrict__ d_C_rho,
                           double D0, double DN,
                           int n_row, double eps) {
    __shared__ double s_rho[BLOCK_SYS];
    int tid = threadIdx.x;

    double my_rho = 0.0;
    const int rho_begin = 2;
    const int rho_end   = n_row + 1 - 2;
    for (int k = rho_begin + tid; k < rho_end; k += blockDim.x) {
        double a = d_A_rho[k]; if (a < 0.0) a = -a;
        double c = d_C_rho[k]; if (c < 0.0) c = -c;
        double v = a > c ? a : c;
        if (v > my_rho) my_rho = v;
    }
    s_rho[tid] = my_rho;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_rho[tid + s] > s_rho[tid]) s_rho[tid] = s_rho[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        double rho = s_rho[0];
        int J;
        if (rho == 0.0 || rho >= 0.5) {
            J = n_row - 1;
        } else {
            double abs_D0 = D0 < 0.0 ? -D0 : D0;
            double abs_DN = DN < 0.0 ? -DN : DN;
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

// 7) Unpack DN from right neighbor.
__global__ void k_unpack_DN(double* D, const double* d_D_right_recv, int n_sys, int n_row) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;
    D[(std::size_t)(n_row - 1) * n_sys + i] = d_D_right_recv[i];
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
    CUDA_CHECK(cudaMalloc(&d_D_right_send_,   sizeof(double) * n_sys));
    CUDA_CHECK(cudaMalloc(&d_D_right_recv_,   sizeof(double) * n_sys));

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
    if (d_D_right_send_)   cudaFree(d_D_right_send_);
    if (d_D_right_recv_)   cudaFree(d_D_right_recv_);
    if (d_A_rho_)          cudaFree(d_A_rho_);
    if (d_C_rho_)          cudaFree(d_C_rho_);
    if (d_J_)              cudaFree(d_J_);
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

// =============================================================================
//  solve_filtered_v1
// =============================================================================

void FilteredTDMACUDA::solve_filtered_v1(double* d_A, double* d_B, double* d_C, double* d_D) {
    if (nprocs_ == 1) {
        tdma_many_cuda(d_A, d_B, d_C, d_D, n_sys_, n_row_);
        return;
    }

    const dim3 block(BLOCK_SYS);
    const dim3 grid = grid1D(n_sys_);

    // 1) Forward elimination (rows 0,1 normalize + 2..n_row-1) — single launch.
    k_fwd_pass<<<grid, block>>>(d_A, d_B, d_C, d_D, n_sys_, n_row_);

    // 2) Backward substitution (rows n_row-3..1) — single launch.
    k_bwd_pass<<<grid, block>>>(d_A, d_C, d_D, n_sys_, n_row_);

    // 3) Pack reduced system
    k_pack<<<grid, block>>>(d_A, d_C, d_D, n_sys_);

    // 4) Pack last row of C and D into dedicated contiguous send buffers,
    //    then exchange with neighbors. Mirrors PaScaL_TDMA's CUDA-aware MPI
    //    fast path: pack kernel → cudaStreamSynchronize → MPI_Isend/Irecv on
    //    plain MPI_DOUBLE contiguous device buffers → MPI_Waitall.
    k_pack_lastrow<<<grid, block>>>(d_C_lastrow_send_, d_C, n_sys_, n_row_);
    k_pack_lastrow<<<grid, block>>>(d_D_lastrow_send_, d_D, n_sys_, n_row_);
    CUDA_CHECK(cudaStreamSynchronize(0));
    {
        MPI_Request req[4];
        MPI_Isend(d_C_lastrow_send_, n_sys_, MPI_DOUBLE, right_rank_, 1, comm_, &req[0]);
        MPI_Irecv(d_C_left_recv_,    n_sys_, MPI_DOUBLE, left_rank_,  1, comm_, &req[1]);
        MPI_Isend(d_D_lastrow_send_, n_sys_, MPI_DOUBLE, right_rank_, 2, comm_, &req[2]);
        MPI_Irecv(d_D_left_recv_,    n_sys_, MPI_DOUBLE, left_rank_,  2, comm_, &req[3]);
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
    }

    // 5) Solve D0 from left boundary (writes d_D_right_send_)
    if (left_rank_ != MPI_PROC_NULL) {
        k_solve_D0_left<<<grid, block>>>(d_D, d_D_right_send_,
                                         d_C_left_recv_, d_D_left_recv_,
                                         d_A, n_sys_);
    }

    // 6) Exchange DN: d_D_right_send_ is already a dedicated buffer.
    CUDA_CHECK(cudaStreamSynchronize(0));
    {
        MPI_Request req[2];
        MPI_Isend(d_D_right_send_, n_sys_, MPI_DOUBLE, left_rank_,  3, comm_, &req[0]);
        MPI_Irecv(d_D_right_recv_, n_sys_, MPI_DOUBLE, right_rank_, 3, comm_, &req[1]);
        MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
    }

    // 7) Unpack DN from right neighbor
    if (right_rank_ != MPI_PROC_NULL) {
        k_unpack_DN<<<grid, block>>>(d_D, d_D_right_recv_, n_sys_, n_row_);
    }

    // 8) Final corrections — single launch covers both left and right ranges.
    const double* D0_dev = d_D;
    const double* DN_dev = d_D + (std::size_t)(n_row_ - 1) * n_sys_;
    int J = cal_J_v1(D0_dev, DN_dev);
    const int j_left_end  = J;
    const int j_right_beg = (n_row_ - 1) - J;
    k_final_pass<<<grid, block>>>(d_A, d_C, d_D, D0_dev, DN_dev,
                                  n_sys_, n_row_, j_left_end, j_right_beg);
}

// =============================================================================
//  solve_filtered_v2
// =============================================================================

void FilteredTDMACUDA::solve_filtered_v2(double* d_A, double* d_B, double* d_C, double* d_D) {
    if (nprocs_ == 1) {
        tdma_many_cuda(d_A, d_B, d_C, d_D, n_sys_, n_row_);
        return;
    }

    const dim3 block(BLOCK_SYS);
    const dim3 grid = grid1D(n_sys_);

    // J is computed up-front from a conservative bound on D0/DN (=2.0 each).
    const int J  = cal_J_v2(2.0, 2.0);
    const int lo = (n_row_ - 2) - J;

    // 1) Forward elimination — full (j≤J) + skip-A (j>J), single kernel.
    k_fwd_pass_v2<<<grid, block>>>(d_A, d_B, d_C, d_D, n_sys_, n_row_, J);

    // 2) Backward substitution — 3 phases, single kernel.
    k_bwd_pass_v2<<<grid, block>>>(d_A, d_C, d_D, n_sys_, n_row_, J, lo);

    // 3) Pack reduced system
    k_pack<<<grid, block>>>(d_A, d_C, d_D, n_sys_);

    // 4) Pack last row into dedicated contiguous send buffers, then exchange.
    //    Same CUDA-aware-MPI fast path as v1 / PaScaL_TDMA.
    k_pack_lastrow<<<grid, block>>>(d_C_lastrow_send_, d_C, n_sys_, n_row_);
    k_pack_lastrow<<<grid, block>>>(d_D_lastrow_send_, d_D, n_sys_, n_row_);
    CUDA_CHECK(cudaStreamSynchronize(0));
    {
        MPI_Request req[4];
        MPI_Isend(d_C_lastrow_send_, n_sys_, MPI_DOUBLE, right_rank_, 1, comm_, &req[0]);
        MPI_Irecv(d_C_left_recv_,    n_sys_, MPI_DOUBLE, left_rank_,  1, comm_, &req[1]);
        MPI_Isend(d_D_lastrow_send_, n_sys_, MPI_DOUBLE, right_rank_, 2, comm_, &req[2]);
        MPI_Irecv(d_D_left_recv_,    n_sys_, MPI_DOUBLE, left_rank_,  2, comm_, &req[3]);
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
    }

    // 5) D0 from left
    if (left_rank_ != MPI_PROC_NULL) {
        k_solve_D0_left<<<grid, block>>>(d_D, d_D_right_send_,
                                         d_C_left_recv_, d_D_left_recv_,
                                         d_A, n_sys_);
    }

    // 6) Exchange DN
    CUDA_CHECK(cudaStreamSynchronize(0));
    {
        MPI_Request req[2];
        MPI_Isend(d_D_right_send_, n_sys_, MPI_DOUBLE, left_rank_,  3, comm_, &req[0]);
        MPI_Irecv(d_D_right_recv_, n_sys_, MPI_DOUBLE, right_rank_, 3, comm_, &req[1]);
        MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
    }

    // 7) Unpack DN
    if (right_rank_ != MPI_PROC_NULL) {
        k_unpack_DN<<<grid, block>>>(d_D, d_D_right_recv_, n_sys_, n_row_);
    }

    // 8) Final corrections (same merged kernel as v1).
    const double* D0_dev = d_D;
    const double* DN_dev = d_D + (std::size_t)(n_row_ - 1) * n_sys_;
    const int j_left_end  = J;
    const int j_right_beg = lo + 1;
    k_final_pass<<<grid, block>>>(d_A, d_C, d_D, D0_dev, DN_dev,
                                  n_sys_, n_row_, j_left_end, j_right_beg);
    (void)d_B;
}
