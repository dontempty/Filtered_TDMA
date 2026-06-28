#include "../include/filtered_tdma_cuda.hpp"
#include "filtered_tdma_kernels.cuh"
#include "ftdma_local_cuda.cuh"

#include <cuda_runtime.h>
#include <mpi.h>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

// =============================================================================
//  Profiling variants of the Filtered TDMA GPU solves.
//
//  Same numerics as the production solves in filtered_tdma_cuda.cu, but with
//  cudaEvent GPU timing (pre-comm vs post-comm kernel groups) and MPI_Wtime
//  communication timing. Results are read via last_gpu_ms() / last_comm_ms().
//
//  Kernels come from filtered_tdma_kernels.cuh (static __global__); each TU
//  gets its own device-code copy so no -rdc is required.
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

// =============================================================================
//  solve_filtered_v1_profile / solve_cycl_filtered_v1_profile
// =============================================================================

void FilteredTDMACUDA::solve_filtered_v1_profile(double* d_A, double* d_B, double* d_C, double* d_D) {
    if (nprocs_ == 1) { ftdma_many_cuda(d_A, d_B, d_C, d_D, n_sys_, n_row_); return; }

    const dim3 block(FTDMA_BLOCK_SYS);
    const dim3 grid = ftdma_grid1D(n_sys_);

    // PRE: fwd/bwd passes + pack for MPI send
    cudaEventRecord(e_gpu_start_);
    k_fwd_pass_v1<<<grid, block>>>(d_A, d_B, d_C, d_D, n_sys_, n_row_);
    k_bwd_pass_v1<<<grid, block>>>(d_A, d_C, d_D, n_sys_, n_row_);
    k_pack_lastrow<<<grid, block>>>(d_send_right_, d_C, d_D, n_sys_, n_row_);
    k_pack_row0<<<grid, block>>>(d_send_left_, d_A, d_D, n_sys_);
    cudaEventRecord(e_gpu_pre_end_);

    CUDA_CHECK(cudaStreamSynchronize(0));
    double t0 = MPI_Wtime();
    mpi_exchange_();
    last_comm_ms_ = (MPI_Wtime() - t0) * 1e3;

    // POST: interface solve + J cutoff + final correction
    cudaEventRecord(e_gpu_post_start_);
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
    cudaEventRecord(e_gpu_post_end_);

    cudaEventSynchronize(e_gpu_post_end_);
    float ms_pre = 0.0f, ms_post = 0.0f;
    cudaEventElapsedTime(&ms_pre,  e_gpu_start_,      e_gpu_pre_end_);
    cudaEventElapsedTime(&ms_post, e_gpu_post_start_,  e_gpu_post_end_);
    last_gpu_ms_ = ms_pre + ms_post;
}

void FilteredTDMACUDA::solve_cycl_filtered_v1_profile(double* d_A, double* d_B, double* d_C, double* d_D) {
    if (nprocs_ == 1) {
        if (!d_E_) CUDA_CHECK(cudaMalloc(&d_E_, sizeof(double) * (std::size_t)n_sys_ * n_row_));
        ftdma_cyclic_many_cuda(d_A, d_B, d_C, d_D, d_E_, n_sys_, n_row_);
        return;
    }

    const dim3 block(FTDMA_BLOCK_SYS);
    const dim3 grid = ftdma_grid1D(n_sys_);

    cudaEventRecord(e_gpu_start_);
    k_fwd_pass_v1<<<grid, block>>>(d_A, d_B, d_C, d_D, n_sys_, n_row_);
    k_bwd_pass_v1<<<grid, block>>>(d_A, d_C, d_D, n_sys_, n_row_);
    k_pack_lastrow<<<grid, block>>>(d_send_right_, d_C, d_D, n_sys_, n_row_);
    k_pack_row0<<<grid, block>>>(d_send_left_, d_A, d_D, n_sys_);
    cudaEventRecord(e_gpu_pre_end_);

    CUDA_CHECK(cudaStreamSynchronize(0));
    double t0 = MPI_Wtime();
    mpi_exchange_();
    last_comm_ms_ = (MPI_Wtime() - t0) * 1e3;

    cudaEventRecord(e_gpu_post_start_);
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
    cudaEventRecord(e_gpu_post_end_);

    cudaEventSynchronize(e_gpu_post_end_);
    float ms_pre = 0.0f, ms_post = 0.0f;
    cudaEventElapsedTime(&ms_pre,  e_gpu_start_,      e_gpu_pre_end_);
    cudaEventElapsedTime(&ms_post, e_gpu_post_start_,  e_gpu_post_end_);
    last_gpu_ms_ = ms_pre + ms_post;
}

// =============================================================================
//  solve_filtered_v2_profile / solve_cycl_filtered_v2_profile
// =============================================================================

void FilteredTDMACUDA::solve_filtered_v2_profile(double* d_A, double* d_B, double* d_C, double* d_D) {
    if (nprocs_ == 1) { ftdma_many_cuda(d_A, d_B, d_C, d_D, n_sys_, n_row_); return; }

    const dim3 block(FTDMA_BLOCK_SYS);
    const dim3 grid = ftdma_grid1D(n_sys_);

    // PRE: compute J, filtered fwd/bwd passes, pack
    cudaEventRecord(e_gpu_start_);
    const double* D0_pre = d_D;
    const double* DN_pre = d_D + (std::size_t)(n_row_ - 1) * n_sys_;
    k_cal_J_rhs_bound<<<1, FTDMA_BLOCK_SYS>>>(d_J_, d_A_rho_, d_C_rho_,
                                               D0_pre, DN_pre, n_sys_, n_row_, eps_);
    k_fwd_pass_v2<<<grid, block>>>(d_A, d_B, d_C, d_D, n_sys_, n_row_, d_J_);
    k_bwd_pass_v2<<<grid, block>>>(d_A, d_C, d_D, n_sys_, n_row_, d_J_);
    k_pack_lastrow<<<grid, block>>>(d_send_right_, d_C, d_D, n_sys_, n_row_);
    k_pack_row0<<<grid, block>>>(d_send_left_, d_A, d_D, n_sys_);
    cudaEventRecord(e_gpu_pre_end_);

    CUDA_CHECK(cudaStreamSynchronize(0));
    double t0 = MPI_Wtime();
    mpi_exchange_();
    last_comm_ms_ = (MPI_Wtime() - t0) * 1e3;

    // POST: interface solve + final correction (J already in d_J_ from PRE)
    cudaEventRecord(e_gpu_post_start_);
    k_solve_both<<<grid, block>>>(d_D, d_A, d_C,
                                  d_recv_left_,  d_recv_left_  + n_sys_,
                                  d_recv_right_, d_recv_right_ + n_sys_,
                                  (left_rank_  != MPI_PROC_NULL) ? 1 : 0,
                                  (right_rank_ != MPI_PROC_NULL) ? 1 : 0,
                                  n_sys_, n_row_);
    const double* D0_dev = d_D;
    const double* DN_dev = d_D + (std::size_t)(n_row_ - 1) * n_sys_;
    k_final_pass<<<grid, block>>>(d_A, d_C, d_D, D0_dev, DN_dev, n_sys_, n_row_, d_J_);
    cudaEventRecord(e_gpu_post_end_);

    cudaEventSynchronize(e_gpu_post_end_);
    float ms_pre = 0.0f, ms_post = 0.0f;
    cudaEventElapsedTime(&ms_pre,  e_gpu_start_,      e_gpu_pre_end_);
    cudaEventElapsedTime(&ms_post, e_gpu_post_start_,  e_gpu_post_end_);
    last_gpu_ms_ = ms_pre + ms_post;
}

void FilteredTDMACUDA::solve_cycl_filtered_v2_profile(double* d_A, double* d_B, double* d_C, double* d_D) {
    if (nprocs_ == 1) {
        if (!d_E_) CUDA_CHECK(cudaMalloc(&d_E_, sizeof(double) * (std::size_t)n_sys_ * n_row_));
        ftdma_cyclic_many_cuda(d_A, d_B, d_C, d_D, d_E_, n_sys_, n_row_);
        return;
    }

    const dim3 block(FTDMA_BLOCK_SYS);
    const dim3 grid = ftdma_grid1D(n_sys_);

    cudaEventRecord(e_gpu_start_);
    const double* D0_pre = d_D;
    const double* DN_pre = d_D + (std::size_t)(n_row_ - 1) * n_sys_;
    k_cal_J_rhs_bound<<<1, FTDMA_BLOCK_SYS>>>(d_J_, d_A_rho_, d_C_rho_,
                                               D0_pre, DN_pre, n_sys_, n_row_, eps_);
    k_fwd_pass_v2<<<grid, block>>>(d_A, d_B, d_C, d_D, n_sys_, n_row_, d_J_);
    k_bwd_pass_v2<<<grid, block>>>(d_A, d_C, d_D, n_sys_, n_row_, d_J_);
    k_pack_lastrow<<<grid, block>>>(d_send_right_, d_C, d_D, n_sys_, n_row_);
    k_pack_row0<<<grid, block>>>(d_send_left_, d_A, d_D, n_sys_);
    cudaEventRecord(e_gpu_pre_end_);

    CUDA_CHECK(cudaStreamSynchronize(0));
    double t0 = MPI_Wtime();
    mpi_exchange_();
    last_comm_ms_ = (MPI_Wtime() - t0) * 1e3;

    cudaEventRecord(e_gpu_post_start_);
    k_solve_both<<<grid, block>>>(d_D, d_A, d_C,
                                  d_recv_left_,  d_recv_left_  + n_sys_,
                                  d_recv_right_, d_recv_right_ + n_sys_,
                                  (left_rank_  != MPI_PROC_NULL) ? 1 : 0,
                                  (right_rank_ != MPI_PROC_NULL) ? 1 : 0,
                                  n_sys_, n_row_);
    const double* D0_dev = d_D;
    const double* DN_dev = d_D + (std::size_t)(n_row_ - 1) * n_sys_;
    k_final_pass<<<grid, block>>>(d_A, d_C, d_D, D0_dev, DN_dev, n_sys_, n_row_, d_J_);
    cudaEventRecord(e_gpu_post_end_);

    cudaEventSynchronize(e_gpu_post_end_);
    float ms_pre = 0.0f, ms_post = 0.0f;
    cudaEventElapsedTime(&ms_pre,  e_gpu_start_,      e_gpu_pre_end_);
    cudaEventElapsedTime(&ms_post, e_gpu_post_start_,  e_gpu_post_end_);
    last_gpu_ms_ = ms_pre + ms_post;
}
