#include "../include/filtered_tdma.hpp"
#include "filtered_tdma_kernels.hpp"
#include "tdma_local.hpp"

#include <cstddef>
#include <cmath>
#include <algorithm>
#include <vector>

using namespace ftdma_kernel;

// ============================================================================
//  Constructor / Destructor
// ============================================================================

FilteredTDMA::FilteredTDMA(int n_sys, int n_row,
                           int myrank, int nprocs, MPI_Comm comm,
                           int left_rank, int right_rank,
                           double eps_constant)
    : comm_(comm), n_sys_(n_sys), n_row_(n_row), nprocs_(nprocs),
      left_rank_(left_rank), right_rank_(right_rank)
{
    // MPI subarray types for boundary-row communication
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

    // Receive buffers
    C_left_recv_.resize(n_sys);
    D_left_recv_.resize(n_sys);

    // Single bidirectional exchange buffers (right neighbor's row 0).
    A_right_recv_.resize(n_sys);
    D0_right_recv_.resize(n_sys);

    // Spectral radius arrays (+1 for safe indexing from solve_theta)
    A_rho_.resize(n_row + 1);
    C_rho_.resize(n_row + 1);

    // Epsilon threshold
    int N = n_row * nprocs;
    eps_ = eps_constant / ((double)N * N);
}

FilteredTDMA::~FilteredTDMA() {
    if (row0_type_ != MPI_DATATYPE_NULL) MPI_Type_free(&row0_type_);
    if (rowN_type_ != MPI_DATATYPE_NULL) MPI_Type_free(&rowN_type_);
}

// ============================================================================
//  Cutoff-J estimators
// ============================================================================

int FilteredTDMA::cal_J_rhs_bound(const double* D) {
    const int m = static_cast<int>(A_rho_.size());

    double rho = 0.0;
    #pragma omp simd reduction(max: rho)
    for (int k = 1; k < m-1; ++k) {
        double v = std::max(std::abs(A_rho_[k]), std::abs(C_rho_[k]));
        if (v > rho) rho = v;
    }

    if (rho == 0.0 || rho >= 0.5) return n_row_ - 1;

    double max_b = 0.0;
    const std::size_t total = (std::size_t)n_row_ * n_sys_;
    for (std::size_t idx = 0; idx < total; ++idx) {
        double v = std::abs(D[idx]);
        if (v > max_b) max_b = v;
    }

    double lambda_p = (1.0 + std::sqrt(1.0 - 4.0 * rho * rho)) / 2.0;
    double q = rho / lambda_p;
    double B = q * (2.0 + q) / ((1.0 - q) * (1.0 - 2.0 * rho)) * max_b;
    int J = static_cast<int>(std::log(eps_ / B) / std::log(q)) + 1;
    return std::min(J, n_row_ - 1);
}

int FilteredTDMA::cal_J_v1(const double* D0, const double* DN) {
    const int m = static_cast<int>(A_rho_.size());

    double rho = 0.0;
    #pragma omp simd reduction(max: rho)
    for (int k = 1; k < m-1; ++k) {
        double v = std::max(std::abs(A_rho_[k]), std::abs(C_rho_[k]));
        if (v > rho) rho = v;
    }

    double max_D0 = 0.0, max_DN = 0.0;
    #pragma omp simd reduction(max: max_D0, max_DN)
    for (int i = 0; i < n_sys_; ++i) {
        double ad0 = std::abs(D0[i]);
        double adn = std::abs(DN[i]);
        if (ad0 > max_D0) max_D0 = ad0;
        if (adn > max_DN) max_DN = adn;
    }

    if (rho == 0.0 || rho >= 0.5) return n_row_ - 1;

    double lambda_p = (1.0 + std::sqrt(1.0 - 4.0 * rho * rho)) / 2.0;
    double q = rho / lambda_p;
    double K = (max_D0 + max_DN) * q / (1.0 - q);
    int J = static_cast<int>(std::log(eps_ / K) / std::log(q)) + 1;
    return std::min(J, n_row_ - 1);
}

int FilteredTDMA::cal_J_v2(double D0, double DN) {
    const int m = static_cast<int>(A_rho_.size());

    double rho = 0.0;
    #pragma omp simd reduction(max: rho)
    for (int k = 1; k < m-1; ++k) {
        double v = std::max(std::abs(A_rho_[k]), std::abs(C_rho_[k]));
        if (v > rho) rho = v;
    }

    // rho >= 0.5: sqrt(1-4*rho^2) would be imaginary → use full propagation
    // rho == 0: A_rho_ not set or truly diagonal → full propagation is safe
    if (rho == 0.0 || rho >= 0.5) return n_row_ - 1;

    double abs_D0 = std::abs(D0);
    double abs_DN = std::abs(DN);

    double lambda_p = (1.0 + std::sqrt(1.0 - 4.0 * rho * rho)) / 2.0;
    double q = rho / lambda_p;
    double B = (abs_D0 + abs_DN) * q / (1.0 - q) + abs_D0 * q * q / (1.0 - q);
    int J = static_cast<int>(std::log(eps_ / B) / std::log(q)) + 1;
    return std::min(J, n_row_ - 1);
}

// ============================================================================
//  MPI neighbour exchange (one bidirectional comm round)
//    my row N-1 (C,D) -> right ; my row 0 (A,D) -> left
//    recv left's row N-1 (LEFT 2x2) and right's row 0 (RIGHT 2x2).
//    The reduced system decouples into independent interface 2x2 blocks, so
//    each rank solves BOTH its interfaces locally (no back-communication).
//    For a PERIODIC communicator every rank has both neighbours, so the
//    wrap-around interface (rank N-1 <-> rank 0) is just another interface.
// ============================================================================

void FilteredTDMA::exchange_interfaces_(double* A, double* C, double* D) {
    MPI_Request req[8];
    int nreq = 0;
    MPI_Isend(C, 1, rowN_type_,             right_rank_, 1, comm_, &req[nreq++]);
    MPI_Irecv(C_left_recv_.data(),   n_sys_, MPI_DOUBLE, left_rank_,  1, comm_, &req[nreq++]);
    MPI_Isend(D, 1, rowN_type_,             right_rank_, 2, comm_, &req[nreq++]);
    MPI_Irecv(D_left_recv_.data(),   n_sys_, MPI_DOUBLE, left_rank_,  2, comm_, &req[nreq++]);
    MPI_Isend(A, n_sys_, MPI_DOUBLE,         left_rank_,  3, comm_, &req[nreq++]);
    MPI_Irecv(A_right_recv_.data(),  n_sys_, MPI_DOUBLE, right_rank_, 3, comm_, &req[nreq++]);
    MPI_Isend(D, n_sys_, MPI_DOUBLE,         left_rank_,  4, comm_, &req[nreq++]);
    MPI_Irecv(D0_right_recv_.data(), n_sys_, MPI_DOUBLE, right_rank_, 4, comm_, &req[nreq++]);
    MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
}

// ============================================================================
//  solve_filtered_v1 / solve_cycl_filtered_v1 — no timing
//  v1: full forward/backward sweep, J cutoff applied only to the final solve.
// ============================================================================

void FilteredTDMA::solve_filtered_v1(double* __restrict A, double* __restrict B,
                                     double* __restrict C, double* __restrict D) {
    if (nprocs_ == 1) { ftdma_many(A, B, C, D, n_sys_, n_row_); return; }

    fwd_elim_v1(A, B, C, D, n_sys_, n_row_);
    bwd_subst_v1(A, C, D, n_sys_, n_row_);
    pack_reduced(A, C, D, n_sys_);
    exchange_interfaces_(A, C, D);
    solve_interfaces(A, C, D, C_left_recv_.data(), D_left_recv_.data(),
                     A_right_recv_.data(), D0_right_recv_.data(),
                     left_rank_  != MPI_PROC_NULL,
                     right_rank_ != MPI_PROC_NULL, n_sys_, n_row_);

    const double* DN = D + (std::size_t)(n_row_ - 1) * n_sys_;
    int J = cal_J_v1(D, DN);
    int left_end  = std::min(J, n_row_ - 2);
    int right_beg = std::max(1, (n_row_ - 1) - J);
    final_solve(A, C, D, n_sys_, n_row_, left_end, right_beg);
}

void FilteredTDMA::solve_cycl_filtered_v1(double* __restrict A, double* __restrict B,
                                          double* __restrict C, double* __restrict D) {
    // Periodic comm makes both interfaces (incl. the wrap-around) real; the
    // multi-rank body is identical to the non-cyclic solve.
    if (nprocs_ == 1) { ftdma_cyclic_many(A, B, C, D, n_sys_, n_row_); return; }

    fwd_elim_v1(A, B, C, D, n_sys_, n_row_);
    bwd_subst_v1(A, C, D, n_sys_, n_row_);
    pack_reduced(A, C, D, n_sys_);
    exchange_interfaces_(A, C, D);
    solve_interfaces(A, C, D, C_left_recv_.data(), D_left_recv_.data(),
                     A_right_recv_.data(), D0_right_recv_.data(),
                     left_rank_  != MPI_PROC_NULL,
                     right_rank_ != MPI_PROC_NULL, n_sys_, n_row_);

    const double* DN = D + (std::size_t)(n_row_ - 1) * n_sys_;
    int J = cal_J_v1(D, DN);
    int left_end  = std::min(J, n_row_ - 2);
    int right_beg = std::max(1, (n_row_ - 1) - J);
    final_solve(A, C, D, n_sys_, n_row_, left_end, right_beg);
}

// ============================================================================
//  solve_filtered_v2 / solve_cycl_filtered_v2 — no timing
//  v2: J computed up front; forward/backward skip work beyond the cutoff.
// ============================================================================

void FilteredTDMA::solve_filtered_v2(double* __restrict A, double* __restrict B,
                                     double* __restrict C, double* __restrict D) {
    if (nprocs_ == 1) { ftdma_many(A, B, C, D, n_sys_, n_row_); return; }

    const int J  = cal_J_rhs_bound(D);
    const int lo = (n_row_ - 2) - J;

    fwd_elim_v2(A, B, C, D, n_sys_, n_row_, J);
    bwd_subst_v2(A, C, D, n_sys_, n_row_, J, lo);
    pack_reduced(A, C, D, n_sys_);
    exchange_interfaces_(A, C, D);
    solve_interfaces(A, C, D, C_left_recv_.data(), D_left_recv_.data(),
                     A_right_recv_.data(), D0_right_recv_.data(),
                     left_rank_  != MPI_PROC_NULL,
                     right_rank_ != MPI_PROC_NULL, n_sys_, n_row_);

    final_solve(A, C, D, n_sys_, n_row_, /*left_end=*/J, /*right_beg=*/lo + 1);
}

void FilteredTDMA::solve_cycl_filtered_v2(double* __restrict A, double* __restrict B,
                                          double* __restrict C, double* __restrict D) {
    if (nprocs_ == 1) { ftdma_cyclic_many(A, B, C, D, n_sys_, n_row_); return; }

    const int J  = cal_J_rhs_bound(D);
    const int lo = (n_row_ - 2) - J;

    fwd_elim_v2(A, B, C, D, n_sys_, n_row_, J);
    bwd_subst_v2(A, C, D, n_sys_, n_row_, J, lo);
    pack_reduced(A, C, D, n_sys_);
    exchange_interfaces_(A, C, D);
    solve_interfaces(A, C, D, C_left_recv_.data(), D_left_recv_.data(),
                     A_right_recv_.data(), D0_right_recv_.data(),
                     left_rank_  != MPI_PROC_NULL,
                     right_rank_ != MPI_PROC_NULL, n_sys_, n_row_);

    final_solve(A, C, D, n_sys_, n_row_, /*left_end=*/J, /*right_beg=*/lo + 1);
}
