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

// Block-wide reduction for max|D0|, max|DN|.
// Launched with one block; computes d_out[0]=max|D0|, d_out[1]=max|DN|.
__global__ void k_max_abs_D0_DN(double* __restrict__ d_out,
                                const double* __restrict__ d_D0,
                                const double* __restrict__ d_DN,
                                int n_sys) {
    __shared__ double s_m0[BLOCK_SYS];
    __shared__ double s_mN[BLOCK_SYS];
    double m0 = 0.0, mN = 0.0;
    for (int i = threadIdx.x; i < n_sys; i += blockDim.x) {
        double ad0 = fabs(d_D0[i]);
        double adn = fabs(d_DN[i]);
        if (ad0 > m0) m0 = ad0;
        if (adn > mN) mN = adn;
    }
    s_m0[threadIdx.x] = m0;
    s_mN[threadIdx.x] = mN;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (s_m0[threadIdx.x + stride] > s_m0[threadIdx.x]) s_m0[threadIdx.x] = s_m0[threadIdx.x + stride];
            if (s_mN[threadIdx.x + stride] > s_mN[threadIdx.x]) s_mN[threadIdx.x] = s_mN[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        d_out[0] = s_m0[0];
        d_out[1] = s_mN[0];
    }
}

// 1) Forward elimination — rows 0 and 1 normalization.
__global__ void k_fwd_norm_01(double* A, double* B, double* C, double* D, int n_sys) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;
    double r0 = 1.0 / B[i];
    A[i] *= r0; C[i] *= r0; D[i] *= r0;
    double r1 = 1.0 / B[(std::size_t)1 * n_sys + i];
    A[(std::size_t)1 * n_sys + i] *= r1;
    C[(std::size_t)1 * n_sys + i] *= r1;
    D[(std::size_t)1 * n_sys + i] *= r1;
}

// 1) Forward elimination — single row j ≥ 2 (full update including A).
__global__ void k_fwd_step(double* A, double* B, double* C, double* D, int n_sys, int j) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;
    std::size_t off_j  = (std::size_t)j * n_sys + i;
    std::size_t off_jm = (std::size_t)(j - 1) * n_sys + i;
    double Aj = A[off_j],  Bj = B[off_j],  Cj = C[off_j],  Dj = D[off_j];
    double Ajm = A[off_jm], Cjm = C[off_jm], Djm = D[off_jm];
    double inv = 1.0 / (Bj - Aj * Cjm);
    D[off_j] =  inv * (Dj - Aj * Djm);
    C[off_j] =  inv * Cj;
    A[off_j] = -inv * Aj * Ajm;
}

// 1) Forward elimination — single row j ≥ 2 (skip A update; used in v2 phase 2).
__global__ void k_fwd_step_skipA(double* A, double* B, double* C, double* D, int n_sys, int j) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;
    std::size_t off_j  = (std::size_t)j * n_sys + i;
    std::size_t off_jm = (std::size_t)(j - 1) * n_sys + i;
    double Aj = A[off_j],  Bj = B[off_j],  Cj = C[off_j],  Dj = D[off_j];
    double Cjm = C[off_jm], Djm = D[off_jm];
    double inv = 1.0 / (Bj - Aj * Cjm);
    D[off_j] = inv * (Dj - Aj * Djm);
    C[off_j] = inv * Cj;
}

// 2) Backward substitution — single row j with full (D,A,C) update.
__global__ void k_bwd_step(double* A, double* C, double* D, int n_sys, int j) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;
    std::size_t off_j  = (std::size_t)j * n_sys + i;
    std::size_t off_jp = (std::size_t)(j + 1) * n_sys + i;
    double Cj = C[off_j];
    double Djp = D[off_jp], Ajp = A[off_jp], Cjp = C[off_jp];
    D[off_j] -= Cj * Djp;
    A[off_j] -= Cj * Ajp;
    C[off_j]  = -Cj * Cjp;
}

// 2) v2 backward — D-only update at row j.
__global__ void k_bwd_D_only(double* C, double* D, int n_sys, int j) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;
    std::size_t off_j  = (std::size_t)j * n_sys + i;
    std::size_t off_jp = (std::size_t)(j + 1) * n_sys + i;
    D[off_j] -= C[off_j] * D[off_jp];
}

// 2) v2 backward — A-only update at row j.
__global__ void k_bwd_A_only(double* A, double* C, int n_sys, int j) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;
    std::size_t off_j  = (std::size_t)j * n_sys + i;
    std::size_t off_jp = (std::size_t)(j + 1) * n_sys + i;
    A[off_j] -= C[off_j] * A[off_jp];
}

// 2) v2 backward — C *= -C[j+1] update at row j.
__global__ void k_bwd_Cmul(double* C, int n_sys, int j) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;
    std::size_t off_j  = (std::size_t)j * n_sys + i;
    std::size_t off_jp = (std::size_t)(j + 1) * n_sys + i;
    C[off_j] *= -C[off_jp];
}

// 3) Pack reduced system — combine rows 0 and 1 (decouple row 0 from x_{N-1}).
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

// 7) Unpack DN from right neighbor.
__global__ void k_unpack_DN(double* D, const double* d_D_right_recv, int n_sys, int n_row) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;
    D[(std::size_t)(n_row - 1) * n_sys + i] = d_D_right_recv[i];
}

// 8a) Left correction — Dj -= Aj * D0 at row j (1 ≤ j ≤ j_left_end).
__global__ void k_final_left(double* A, double* D, const double* D0, int n_sys, int j) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;
    std::size_t off = (std::size_t)j * n_sys + i;
    D[off] -= A[off] * D0[i];
}

// 8b) Right correction — Dj -= Cj * DN at row j (jrb ≤ j ≤ n_row-2).
__global__ void k_final_right(double* C, double* D, const double* DN, int n_sys, int j) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;
    std::size_t off = (std::size_t)j * n_sys + i;
    D[off] -= C[off] * DN[i];
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
    (void)myrank;  // matches CPU API; not stored separately

    // MPI subarray types for boundary-row communication (identical to CPU)
    std::vector<int> bigsize = { n_row, n_sys };
    std::vector<int> subsize = {     1, n_sys };
    std::vector<int> start_0 = {     0,     0 };
    std::vector<int> start_N = { n_row - 1, 0 };

    MPI_Type_create_subarray(2, bigsize.data(), subsize.data(), start_0.data(),
                             MPI_ORDER_C, MPI_DOUBLE, &row0_type_);
    MPI_Type_commit(&row0_type_);

    MPI_Type_create_subarray(2, bigsize.data(), subsize.data(), start_N.data(),
                             MPI_ORDER_C, MPI_DOUBLE, &rowN_type_);
    MPI_Type_commit(&rowN_type_);

    // Device buffers
    CUDA_CHECK(cudaMalloc(&d_C_left_recv_,  sizeof(double) * n_sys));
    CUDA_CHECK(cudaMalloc(&d_D_left_recv_,  sizeof(double) * n_sys));
    CUDA_CHECK(cudaMalloc(&d_D_right_send_, sizeof(double) * n_sys));
    CUDA_CHECK(cudaMalloc(&d_D_right_recv_, sizeof(double) * n_sys));

    CUDA_CHECK(cudaMalloc(&d_A_rho_, sizeof(double) * (n_row + 1)));
    CUDA_CHECK(cudaMalloc(&d_C_rho_, sizeof(double) * (n_row + 1)));
    CUDA_CHECK(cudaMemset(d_A_rho_, 0, sizeof(double) * (n_row + 1)));
    CUDA_CHECK(cudaMemset(d_C_rho_, 0, sizeof(double) * (n_row + 1)));

    CUDA_CHECK(cudaMalloc(&d_maxD_, sizeof(double) * 2));

    h_A_rho_.assign(n_row + 1, 0.0);
    h_C_rho_.assign(n_row + 1, 0.0);

    // Epsilon threshold
    int N = n_row * nprocs;
    eps_ = eps_constant / ((double)N * N);
}

FilteredTDMACUDA::~FilteredTDMACUDA() {
    if (row0_type_ != MPI_DATATYPE_NULL) MPI_Type_free(&row0_type_);
    if (rowN_type_ != MPI_DATATYPE_NULL) MPI_Type_free(&rowN_type_);
    if (d_C_left_recv_)   cudaFree(d_C_left_recv_);
    if (d_D_left_recv_)   cudaFree(d_D_left_recv_);
    if (d_D_right_send_)  cudaFree(d_D_right_send_);
    if (d_D_right_recv_)  cudaFree(d_D_right_recv_);
    if (d_A_rho_)         cudaFree(d_A_rho_);
    if (d_C_rho_)         cudaFree(d_C_rho_);
    if (d_maxD_)          cudaFree(d_maxD_);
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
    // 1) Pull A_rho, C_rho to host (small: n_row+1 doubles)
    CUDA_CHECK(cudaMemcpy(h_A_rho_.data(), d_A_rho_,
                          sizeof(double) * (n_row_ + 1), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_rho_.data(), d_C_rho_,
                          sizeof(double) * (n_row_ + 1), cudaMemcpyDeviceToHost));

    const int skip = 2;
    const int rho_begin = skip;
    const int rho_end   = static_cast<int>(h_A_rho_.size()) - skip;

    double rho = 0.0;
    for (int k = rho_begin; k < rho_end; ++k) {
        double v = std::max(std::abs(h_A_rho_[k]), std::abs(h_C_rho_[k]));
        if (v > rho) rho = v;
    }

    // 2) Device-side reduction for max|D0|, max|DN|
    k_max_abs_D0_DN<<<1, BLOCK_SYS>>>(d_maxD_, D0_dev, DN_dev, n_sys_);
    double h_maxD[2];
    CUDA_CHECK(cudaMemcpy(h_maxD, d_maxD_, sizeof(double) * 2, cudaMemcpyDeviceToHost));
    double max_D0 = h_maxD[0];
    double max_DN = h_maxD[1];

    if (rho == 0.0 || rho >= 0.5) return n_row_ - 1;

    double lambda_p = (1.0 + std::sqrt(1.0 - 4.0 * rho * rho)) / 2.0;
    double q = rho / lambda_p;
    double K = (max_D0 + max_DN) * q / (1.0 - q);
    int J = static_cast<int>(std::log(eps_ / K) / std::log(q)) + 1;
    return std::min(J, n_row_ - 1);
}

int FilteredTDMACUDA::cal_J_v2(double D0, double DN) {
    CUDA_CHECK(cudaMemcpy(h_A_rho_.data(), d_A_rho_,
                          sizeof(double) * (n_row_ + 1), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_rho_.data(), d_C_rho_,
                          sizeof(double) * (n_row_ + 1), cudaMemcpyDeviceToHost));

    const int skip = 2;
    const int rho_begin = skip;
    const int rho_end   = static_cast<int>(h_A_rho_.size()) - skip;

    double rho = 0.0;
    for (int k = rho_begin; k < rho_end; ++k) {
        double v = std::max(std::abs(h_A_rho_[k]), std::abs(h_C_rho_[k]));
        if (v > rho) rho = v;
    }
    if (rho == 0.0 || rho >= 0.5) return n_row_ - 1;

    double abs_D0 = std::abs(D0);
    double abs_DN = std::abs(DN);
    double lambda_p = (1.0 + std::sqrt(1.0 - 4.0 * rho * rho)) / 2.0;
    double q = rho / lambda_p;
    double B = (abs_D0 + abs_DN) * q / (1.0 - q) + abs_D0 * q * q / (1.0 - q);
    int J = static_cast<int>(std::log(eps_ / B) / std::log(q)) + 1;
    return std::min(J, n_row_ - 1);
}

// =============================================================================
//  solve_filtered_v1
// =============================================================================

void FilteredTDMACUDA::solve_filtered_v1(double* d_A, double* d_B, double* d_C, double* d_D) {
    if (nprocs_ == 1) {
        // Single-rank fast path: standard sequential Thomas on device.
        tdma_many_cuda(d_A, d_B, d_C, d_D, n_sys_, n_row_);
        return;
    }

    const dim3 block(BLOCK_SYS);
    const dim3 grid = grid1D(n_sys_);

    // 1) Forward Elimination
    k_fwd_norm_01<<<grid, block>>>(d_A, d_B, d_C, d_D, n_sys_);
    for (int j = 2; j < n_row_; ++j) {
        k_fwd_step<<<grid, block>>>(d_A, d_B, d_C, d_D, n_sys_, j);
    }

    // 2) Backward Substitution
    for (int j = n_row_ - 3; j >= 1; --j) {
        k_bwd_step<<<grid, block>>>(d_A, d_C, d_D, n_sys_, j);
    }

    // 3) Pack reduced system
    k_pack<<<grid, block>>>(d_A, d_C, d_D, n_sys_);

    // 4) Send C[N-1], D[N-1] to right; recv C_left, D_left from left
    CUDA_CHECK(cudaDeviceSynchronize());
    {
        MPI_Request req[4];
        MPI_Isend(d_C, 1, rowN_type_, right_rank_, 1, comm_, &req[0]);
        MPI_Irecv(d_C_left_recv_, n_sys_, MPI_DOUBLE, left_rank_,  1, comm_, &req[1]);
        MPI_Isend(d_D, 1, rowN_type_, right_rank_, 2, comm_, &req[2]);
        MPI_Irecv(d_D_left_recv_, n_sys_, MPI_DOUBLE, left_rank_,  2, comm_, &req[3]);
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
    }

    // 5) Solve D0 from left boundary; compute left neighbor's DN
    if (left_rank_ != MPI_PROC_NULL) {
        k_solve_D0_left<<<grid, block>>>(d_D, d_D_right_send_,
                                         d_C_left_recv_, d_D_left_recv_,
                                         d_A, n_sys_);
    }

    // 6) Send computed DN to left neighbor; recv own DN from right neighbor
    CUDA_CHECK(cudaDeviceSynchronize());
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

    // 8) Final local solve — compute J (host) then apply corrections
    const double* D0_dev = d_D;
    const double* DN_dev = d_D + (std::size_t)(n_row_ - 1) * n_sys_;
    int J = cal_J_v1(D0_dev, DN_dev);

    const int j_left_end  = J;
    const int j_right_beg = (n_row_ - 1) - J;
    for (int j = 1; j <= j_left_end && j < n_row_ - 1; ++j) {
        k_final_left<<<grid, block>>>(d_A, d_D, D0_dev, n_sys_, j);
    }
    {
        int jrb = std::max(1, j_right_beg);
        for (int j = jrb; j < n_row_ - 1; ++j) {
            k_final_right<<<grid, block>>>(d_C, d_D, DN_dev, n_sys_, j);
        }
    }
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

    const int J  = cal_J_v2(2.0, 2.0);
    const int lo = (n_row_ - 2) - J;

    // 1) Forward Elimination — phase 1 full (j ≤ J), phase 2 skip A (j > J)
    k_fwd_norm_01<<<grid, block>>>(d_A, d_B, d_C, d_D, n_sys_);
    for (int j = 2; j <= J; ++j) {
        k_fwd_step<<<grid, block>>>(d_A, d_B, d_C, d_D, n_sys_, j);
    }
    for (int j = J + 1; j < n_row_; ++j) {
        k_fwd_step_skipA<<<grid, block>>>(d_A, d_B, d_C, d_D, n_sys_, j);
    }

    // 2) Backward Substitution — 3 phases
    for (int j = n_row_ - 3; j >= 1; --j) {
        k_bwd_D_only<<<grid, block>>>(d_C, d_D, n_sys_, j);
    }
    if (J >= 2) {
        for (int j = J - 1; j >= 1; --j) {
            k_bwd_A_only<<<grid, block>>>(d_A, d_C, n_sys_, j);
        }
    }
    for (int j = n_row_ - 3; j >= lo + 1; --j) {
        k_bwd_Cmul<<<grid, block>>>(d_C, n_sys_, j);
    }

    // 3) Pack
    k_pack<<<grid, block>>>(d_A, d_C, d_D, n_sys_);

    // 4) Boundary-row MPI exchange
    CUDA_CHECK(cudaDeviceSynchronize());
    {
        MPI_Request req[4];
        MPI_Isend(d_C, 1, rowN_type_, right_rank_, 1, comm_, &req[0]);
        MPI_Irecv(d_C_left_recv_, n_sys_, MPI_DOUBLE, left_rank_,  1, comm_, &req[1]);
        MPI_Isend(d_D, 1, rowN_type_, right_rank_, 2, comm_, &req[2]);
        MPI_Irecv(d_D_left_recv_, n_sys_, MPI_DOUBLE, left_rank_,  2, comm_, &req[3]);
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
    }

    // 5) D0 from left
    if (left_rank_ != MPI_PROC_NULL) {
        k_solve_D0_left<<<grid, block>>>(d_D, d_D_right_send_,
                                         d_C_left_recv_, d_D_left_recv_,
                                         d_A, n_sys_);
    }

    // 6) Send DN_left, recv DN_right
    CUDA_CHECK(cudaDeviceSynchronize());
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

    // 8) Final local solve
    const double* D0_dev = d_D;
    const double* DN_dev = d_D + (std::size_t)(n_row_ - 1) * n_sys_;
    const int j_left_end  = J;
    const int j_right_beg = lo + 1;
    for (int j = 1; j <= j_left_end && j < n_row_ - 1; ++j) {
        k_final_left<<<grid, block>>>(d_A, d_D, D0_dev, n_sys_, j);
    }
    for (int j = j_right_beg; j < n_row_ - 1; ++j) {
        k_final_right<<<grid, block>>>(d_C, d_D, DN_dev, n_sys_, j);
    }
}
