#include "../include/filtered_tdma.hpp"
#include "filtered_tdma_kernels.hpp"
#include "tdma_local.hpp"

#include <cstddef>
#include <cmath>
#include <algorithm>
#include <vector>

// =============================================================================
//  Per-phase timed variants of the Filtered TDMA solves.
//
//  Same numerics as the production solves in filtered_tdma.cpp, but each step is
//  bracketed by MPI_Barrier + MPI_Wtime so time_list[0..6] records:
//    [0] forward   [1] backward  [2] pack   [3] exchange
//    [4] interface 2x2  [5] (unused, 0)  [6] final local solve
//
//  The algorithm steps are reused from filtered_tdma_kernels.hpp (the same
//  building blocks the production solves call); only the timing scaffold differs.
// =============================================================================

using namespace ftdma_kernel;

// ============================================================================
//  solve_filtered_v1_profile
// ============================================================================

void FilteredTDMA::solve_filtered_v1_profile(double* __restrict A, double* __restrict B,
                                             double* __restrict C, double* __restrict D,
                                             std::vector<double>& time_list) {
    time_list.resize(7);
    if (nprocs_ == 1) {
        double t0 = MPI_Wtime();
        ftdma_many(A, B, C, D, n_sys_, n_row_);
        time_list.assign(7, 0.0);
        time_list[0] = MPI_Wtime() - t0;
        return;
    }

    double t0, t1;

    // 1) Forward elimination
    MPI_Barrier(comm_); t0 = MPI_Wtime();
    fwd_elim_v1(A, B, C, D, n_sys_, n_row_);
    t1 = MPI_Wtime(); time_list[0] = t1 - t0;

    // 2) Backward substitution
    MPI_Barrier(comm_); t0 = MPI_Wtime();
    bwd_subst_v1(A, C, D, n_sys_, n_row_);
    t1 = MPI_Wtime(); time_list[1] = t1 - t0;

    // 3) Pack reduced system
    MPI_Barrier(comm_); t0 = MPI_Wtime();
    pack_reduced(A, C, D, n_sys_);
    t1 = MPI_Wtime(); time_list[2] = t1 - t0;

    // 4) Single bidirectional exchange
    MPI_Barrier(comm_); t0 = MPI_Wtime();
    exchange_interfaces_(A, C, D);
    t1 = MPI_Wtime(); time_list[3] = t1 - t0;

    // 5) Solve both interface 2x2 blocks locally
    MPI_Barrier(comm_); t0 = MPI_Wtime();
    solve_interfaces(A, C, D, C_left_recv_.data(), D_left_recv_.data(),
                     A_right_recv_.data(), D0_right_recv_.data(),
                     left_rank_  != MPI_PROC_NULL,
                     right_rank_ != MPI_PROC_NULL, n_sys_, n_row_);
    t1 = MPI_Wtime(); time_list[4] = t1 - t0;
    time_list[5] = 0.0;

    // 6) Final local solve
    MPI_Barrier(comm_); t0 = MPI_Wtime();
    const double* DN = D + (std::size_t)(n_row_ - 1) * n_sys_;
    int J = cal_J_v1(D, DN);
    int left_end  = std::min(J, n_row_ - 2);
    int right_beg = std::max(1, (n_row_ - 1) - J);
    final_solve(A, C, D, n_sys_, n_row_, left_end, right_beg);
    t1 = MPI_Wtime(); time_list[6] = t1 - t0;
}

// ============================================================================
//  solve_filtered_v2_profile
// ============================================================================

void FilteredTDMA::solve_filtered_v2_profile(double* __restrict A, double* __restrict B,
                                             double* __restrict C, double* __restrict D,
                                             std::vector<double>& time_list) {
    time_list.resize(7);
    if (nprocs_ == 1) {
        double t0 = MPI_Wtime();
        ftdma_many(A, B, C, D, n_sys_, n_row_);
        time_list.assign(7, 0.0);
        time_list[0] = MPI_Wtime() - t0;
        return;
    }

    double t0, t1;
    const int J  = std::min(cal_J_rhs_bound(D), n_row_ - 2);
    const int lo = (n_row_ - 2) - J;

    // 1) Forward elimination (filtered split at J)
    MPI_Barrier(comm_); t0 = MPI_Wtime();
    fwd_elim_v2(A, B, C, D, n_sys_, n_row_, J);
    t1 = MPI_Wtime(); time_list[0] = t1 - t0;

    // 2) Backward substitution (3-phase)
    MPI_Barrier(comm_); t0 = MPI_Wtime();
    bwd_subst_v2(A, C, D, n_sys_, n_row_, J, lo);
    t1 = MPI_Wtime(); time_list[1] = t1 - t0;

    // 3) Pack reduced system
    MPI_Barrier(comm_); t0 = MPI_Wtime();
    pack_reduced(A, C, D, n_sys_);
    t1 = MPI_Wtime(); time_list[2] = t1 - t0;

    // 4) Single bidirectional exchange
    MPI_Barrier(comm_); t0 = MPI_Wtime();
    exchange_interfaces_(A, C, D);
    t1 = MPI_Wtime(); time_list[3] = t1 - t0;

    // 5) Solve both interface 2x2 blocks locally
    MPI_Barrier(comm_); t0 = MPI_Wtime();
    solve_interfaces(A, C, D, C_left_recv_.data(), D_left_recv_.data(),
                     A_right_recv_.data(), D0_right_recv_.data(),
                     left_rank_  != MPI_PROC_NULL,
                     right_rank_ != MPI_PROC_NULL, n_sys_, n_row_);
    t1 = MPI_Wtime(); time_list[4] = t1 - t0;
    time_list[5] = 0.0;

    // 6) Final local solve
    MPI_Barrier(comm_); t0 = MPI_Wtime();
    final_solve(A, C, D, n_sys_, n_row_, /*left_end=*/J,
                                         /*right_beg=*/std::max(1, lo + 1));
    t1 = MPI_Wtime(); time_list[6] = t1 - t0;
}
