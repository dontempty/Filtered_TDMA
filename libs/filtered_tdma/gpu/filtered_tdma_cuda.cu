#include "../include/filtered_tdma_cuda.hpp"
#include "filtered_tdma_kernels.cuh"
#include "ftdma_local_cuda.cuh"

#include <cuda_runtime.h>
#include <mpi.h>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t err = (x);                                                 \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "[CUDA] %s:%d %s -> %s\n", __FILE__, __LINE__,\
                         #x, cudaGetErrorString(err));                         \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

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

    if (nprocs > 1) {
        CUDA_CHECK(cudaMalloc(&d_send_right_, sizeof(double) * 2 * n_sys));
        CUDA_CHECK(cudaMalloc(&d_send_left_,  sizeof(double) * 2 * n_sys));
        CUDA_CHECK(cudaMalloc(&d_recv_left_,  sizeof(double) * 2 * n_sys));
        CUDA_CHECK(cudaMalloc(&d_recv_right_, sizeof(double) * 2 * n_sys));
    }

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
    if (d_A_rho_)           cudaFree(d_A_rho_);
    if (d_C_rho_)           cudaFree(d_C_rho_);
    if (d_J_)               cudaFree(d_J_);
    if (d_E_)               cudaFree(d_E_);
    if (e_gpu_start_)       cudaEventDestroy(e_gpu_start_);
    if (e_gpu_pre_end_)     cudaEventDestroy(e_gpu_pre_end_);
    if (e_gpu_post_start_)  cudaEventDestroy(e_gpu_post_start_);
    if (e_gpu_post_end_)    cudaEventDestroy(e_gpu_post_end_);
}

// =============================================================================
//  set_rho_device
// =============================================================================

void FilteredTDMACUDA::set_rho_device(const double* d_A,
                                      const double* d_B,
                                      const double* d_C) {
    dim3 block(FTDMA_BLOCK_SYS);
    dim3 grid = ftdma_grid1D(n_row_ + 1);
    k_set_rho<<<grid, block>>>(d_A_rho_, d_C_rho_, d_A, d_B, d_C, n_sys_, n_row_);
}

int FilteredTDMACUDA::last_J() const {
    int h_J = -1;
    CUDA_CHECK(cudaMemcpy(&h_J, d_J_, sizeof(int), cudaMemcpyDeviceToHost));
    return h_J;
}

// =============================================================================
//  MPI neighbor exchange (bidirectional, blocking)
// =============================================================================

void FilteredTDMACUDA::mpi_exchange_() {
    MPI_Request req[4];
    int nreq = 0;
    MPI_Isend(d_send_right_, 2 * n_sys_, MPI_DOUBLE, right_rank_, 1, comm_, &req[nreq++]);
    MPI_Irecv(d_recv_left_,  2 * n_sys_, MPI_DOUBLE, left_rank_,  1, comm_, &req[nreq++]);
    MPI_Isend(d_send_left_,  2 * n_sys_, MPI_DOUBLE, left_rank_,  2, comm_, &req[nreq++]);
    MPI_Irecv(d_recv_right_, 2 * n_sys_, MPI_DOUBLE, right_rank_, 2, comm_, &req[nreq++]);
    MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
}

// =============================================================================
//  solve_filtered_v1 / solve_cycl_filtered_v1
// =============================================================================

void FilteredTDMACUDA::solve_filtered_v1(double* d_A, double* d_B, double* d_C, double* d_D) {
    if (nprocs_ == 1) { ftdma_many_cuda(d_A, d_B, d_C, d_D, n_sys_, n_row_); return; }

    const dim3 block(FTDMA_BLOCK_SYS);
    const dim3 grid = ftdma_grid1D(n_sys_);

    k_fwd_pass_v1<<<grid, block>>>(d_A, d_B, d_C, d_D, n_sys_, n_row_);
    k_bwd_pass_v1<<<grid, block>>>(d_A, d_C, d_D, n_sys_, n_row_);
    k_pack_lastrow<<<grid, block>>>(d_send_right_, d_C, d_D, n_sys_, n_row_);
    k_pack_row0<<<grid, block>>>(d_send_left_, d_A, d_D, n_sys_);

    CUDA_CHECK(cudaStreamSynchronize(0));
    mpi_exchange_();

    k_solve_both<<<grid, block>>>(d_D, d_A, d_C,
                                  d_recv_left_,  d_recv_left_  + n_sys_,
                                  d_recv_right_, d_recv_right_ + n_sys_,
                                  (left_rank_  != MPI_PROC_NULL) ? 1 : 0,
                                  (right_rank_ != MPI_PROC_NULL) ? 1 : 0,
                                  n_sys_, n_row_);
    const double* D0_dev = d_D;
    const double* DN_dev = d_D + (std::size_t)(n_row_ - 1) * n_sys_;
    k_cal_J_rhs_bound<<<1, FTDMA_BLOCK_SYS>>>(d_J_, d_A_rho_, d_C_rho_,
                                               D0_dev, DN_dev, n_sys_, n_row_, eps_);
    k_final_pass<<<grid, block>>>(d_A, d_C, d_D, D0_dev, DN_dev, n_sys_, n_row_, d_J_);
}

void FilteredTDMACUDA::solve_cycl_filtered_v1(double* d_A, double* d_B, double* d_C, double* d_D) {
    if (nprocs_ == 1) {
        if (!d_E_) CUDA_CHECK(cudaMalloc(&d_E_, sizeof(double) * (std::size_t)n_sys_ * n_row_));
        ftdma_cyclic_many_cuda(d_A, d_B, d_C, d_D, d_E_, n_sys_, n_row_);
        return;
    }

    const dim3 block(FTDMA_BLOCK_SYS);
    const dim3 grid = ftdma_grid1D(n_sys_);

    k_fwd_pass_v1<<<grid, block>>>(d_A, d_B, d_C, d_D, n_sys_, n_row_);
    k_bwd_pass_v1<<<grid, block>>>(d_A, d_C, d_D, n_sys_, n_row_);
    k_pack_lastrow<<<grid, block>>>(d_send_right_, d_C, d_D, n_sys_, n_row_);
    k_pack_row0<<<grid, block>>>(d_send_left_, d_A, d_D, n_sys_);

    CUDA_CHECK(cudaStreamSynchronize(0));
    mpi_exchange_();

    k_solve_both<<<grid, block>>>(d_D, d_A, d_C,
                                  d_recv_left_,  d_recv_left_  + n_sys_,
                                  d_recv_right_, d_recv_right_ + n_sys_,
                                  (left_rank_  != MPI_PROC_NULL) ? 1 : 0,
                                  (right_rank_ != MPI_PROC_NULL) ? 1 : 0,
                                  n_sys_, n_row_);
    const double* D0_dev = d_D;
    const double* DN_dev = d_D + (std::size_t)(n_row_ - 1) * n_sys_;
    k_cal_J_rhs_bound<<<1, FTDMA_BLOCK_SYS>>>(d_J_, d_A_rho_, d_C_rho_,
                                               D0_dev, DN_dev, n_sys_, n_row_, eps_);
    k_final_pass<<<grid, block>>>(d_A, d_C, d_D, D0_dev, DN_dev, n_sys_, n_row_, d_J_);
}

// =============================================================================
//  solve_filtered_v2 / solve_cycl_filtered_v2
// =============================================================================

void FilteredTDMACUDA::solve_filtered_v2(double* d_A, double* d_B, double* d_C, double* d_D) {
    if (nprocs_ == 1) { ftdma_many_cuda(d_A, d_B, d_C, d_D, n_sys_, n_row_); return; }

    const dim3 block(FTDMA_BLOCK_SYS);
    const dim3 grid = ftdma_grid1D(n_sys_);
    const double* D0_pre = d_D;
    const double* DN_pre = d_D + (std::size_t)(n_row_ - 1) * n_sys_;

    k_cal_J_rhs_bound<<<1, FTDMA_BLOCK_SYS>>>(d_J_, d_A_rho_, d_C_rho_,
                                               D0_pre, DN_pre, n_sys_, n_row_, eps_);
    k_fwd_pass_v2<<<grid, block>>>(d_A, d_B, d_C, d_D, n_sys_, n_row_, d_J_);
    k_bwd_pass_v2<<<grid, block>>>(d_A, d_C, d_D, n_sys_, n_row_, d_J_);
    k_pack_lastrow<<<grid, block>>>(d_send_right_, d_C, d_D, n_sys_, n_row_);
    k_pack_row0<<<grid, block>>>(d_send_left_, d_A, d_D, n_sys_);

    CUDA_CHECK(cudaStreamSynchronize(0));
    mpi_exchange_();

    k_solve_both<<<grid, block>>>(d_D, d_A, d_C,
                                  d_recv_left_,  d_recv_left_  + n_sys_,
                                  d_recv_right_, d_recv_right_ + n_sys_,
                                  (left_rank_  != MPI_PROC_NULL) ? 1 : 0,
                                  (right_rank_ != MPI_PROC_NULL) ? 1 : 0,
                                  n_sys_, n_row_);
    const double* D0_dev = d_D;
    const double* DN_dev = d_D + (std::size_t)(n_row_ - 1) * n_sys_;
    k_final_pass<<<grid, block>>>(d_A, d_C, d_D, D0_dev, DN_dev, n_sys_, n_row_, d_J_);
}

void FilteredTDMACUDA::solve_cycl_filtered_v2(double* d_A, double* d_B, double* d_C, double* d_D) {
    if (nprocs_ == 1) {
        if (!d_E_) CUDA_CHECK(cudaMalloc(&d_E_, sizeof(double) * (std::size_t)n_sys_ * n_row_));
        ftdma_cyclic_many_cuda(d_A, d_B, d_C, d_D, d_E_, n_sys_, n_row_);
        return;
    }

    const dim3 block(FTDMA_BLOCK_SYS);
    const dim3 grid = ftdma_grid1D(n_sys_);
    const double* D0_pre = d_D;
    const double* DN_pre = d_D + (std::size_t)(n_row_ - 1) * n_sys_;

    k_cal_J_rhs_bound<<<1, FTDMA_BLOCK_SYS>>>(d_J_, d_A_rho_, d_C_rho_,
                                               D0_pre, DN_pre, n_sys_, n_row_, eps_);
    k_fwd_pass_v2<<<grid, block>>>(d_A, d_B, d_C, d_D, n_sys_, n_row_, d_J_);
    k_bwd_pass_v2<<<grid, block>>>(d_A, d_C, d_D, n_sys_, n_row_, d_J_);
    k_pack_lastrow<<<grid, block>>>(d_send_right_, d_C, d_D, n_sys_, n_row_);
    k_pack_row0<<<grid, block>>>(d_send_left_, d_A, d_D, n_sys_);

    CUDA_CHECK(cudaStreamSynchronize(0));
    mpi_exchange_();

    k_solve_both<<<grid, block>>>(d_D, d_A, d_C,
                                  d_recv_left_,  d_recv_left_  + n_sys_,
                                  d_recv_right_, d_recv_right_ + n_sys_,
                                  (left_rank_  != MPI_PROC_NULL) ? 1 : 0,
                                  (right_rank_ != MPI_PROC_NULL) ? 1 : 0,
                                  n_sys_, n_row_);
    const double* D0_dev = d_D;
    const double* DN_dev = d_D + (std::size_t)(n_row_ - 1) * n_sys_;
    k_final_pass<<<grid, block>>>(d_A, d_C, d_D, D0_dev, DN_dev, n_sys_, n_row_, d_J_);
}
