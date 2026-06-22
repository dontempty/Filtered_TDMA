#include "filtered_tdma.hpp"
#include "tdma_local.hpp"

#include <cstddef>
#include <cmath>
#include <algorithm>
#include <vector>

struct SolverPtrs {
    double *A0, *A1, *B0, *B1, *C0, *C1, *CN, *D0, *D1, *DN;
};

static inline SolverPtrs setup_ptrs(double* A, double* B, double* C, double* D,
                                     int n_sys, int n_row) {
    SolverPtrs p;
    p.A0 = A;
    p.A1 = A + (std::size_t)1 * n_sys;
    p.B0 = B;
    p.B1 = B + (std::size_t)1 * n_sys;
    p.C0 = C;
    p.C1 = C + (std::size_t)1 * n_sys;
    p.CN = C + (std::size_t)(n_row - 1) * n_sys;
    p.D0 = D;
    p.D1 = D + (std::size_t)1 * n_sys;
    p.DN = D + (std::size_t)(n_row - 1) * n_sys;
    return p;
}

// ============================================================================
//  solve_filtered_v1_profile() — per-phase timing with MPI_Barrier
// ============================================================================

void FilteredTDMA::solve_filtered_v1_profile(double* __restrict A,
                                             double* __restrict B,
                                             double* __restrict C,
                                             double* __restrict D,
                                             std::vector<double>& time_list) {
    time_list.resize(7);
    if (nprocs_ == 1) {
        double t0 = MPI_Wtime();
        tdma_many(A, B, C, D, n_sys_, n_row_);
        time_list.assign(7, 0.0);
        time_list[0] = MPI_Wtime() - t0;
        return;
    }

    int i, j;
    double t0, t1;
    auto p = setup_ptrs(A, B, C, D, n_sys_, n_row_);

    const double* __restrict C_left   = C_left_recv_.data();
    const double* __restrict D_left   = D_left_recv_.data();
    const double* __restrict A_right  = A_right_recv_.data();
    const double* __restrict D0_right = D0_right_recv_.data();

    // 1) Forward Elimination
    MPI_Barrier(comm_); t0 = MPI_Wtime();
    #pragma omp simd
    for (i = 0; i < n_sys_; ++i) {
        double r0 = 1.0 / p.B0[i];
        p.A0[i] *= r0; p.C0[i] *= r0; p.D0[i] *= r0;
        double r1 = 1.0 / p.B1[i];
        p.A1[i] *= r1; p.C1[i] *= r1; p.D1[i] *= r1;
    }
    {
        double* Ajm = p.A1, *Cjm = p.C1, *Djm = p.D1;
        double* Aj = p.A1 + n_sys_, *Bj = p.B1 + n_sys_;
        double* Cj = p.C1 + n_sys_, *Dj = p.D1 + n_sys_;
        for (j = 2; j < n_row_; ++j,
             Ajm = Aj, Cjm = Cj, Djm = Dj,
             Aj += n_sys_, Bj += n_sys_, Cj += n_sys_, Dj += n_sys_) {
            #pragma omp simd
            for (i = 0; i < n_sys_; ++i) {
                double inv = 1.0 / (Bj[i] - Aj[i] * Cjm[i]);
                Dj[i] =  inv * (Dj[i] - Aj[i] * Djm[i]);
                Cj[i] =  inv * Cj[i];
                Aj[i] = -inv * Aj[i] * Ajm[i];
            }
        }
    }
    t1 = MPI_Wtime(); time_list[0] = t1 - t0;

    // 2) Backward Substitution
    MPI_Barrier(comm_); t0 = MPI_Wtime();
    {
        double* Ajp = A + (std::size_t)(n_row_ - 2) * n_sys_;
        double* Cjp = C + (std::size_t)(n_row_ - 2) * n_sys_;
        double* Djp = D + (std::size_t)(n_row_ - 2) * n_sys_;
        double* Aj = Ajp - n_sys_, *Cj = Cjp - n_sys_, *Dj = Djp - n_sys_;
        for (j = n_row_ - 3; j >= 1; --j,
             Ajp = Aj, Cjp = Cj, Djp = Dj,
             Aj -= n_sys_, Cj -= n_sys_, Dj -= n_sys_) {
            #pragma omp simd
            for (i = 0; i < n_sys_; ++i) {
                Dj[i] -= Cj[i] * Djp[i];
                Aj[i] -= Cj[i] * Ajp[i];
                Cj[i] *= -Cjp[i];
            }
        }
    }
    t1 = MPI_Wtime(); time_list[1] = t1 - t0;

    // 3) Pack — C0 = 0
    MPI_Barrier(comm_); t0 = MPI_Wtime();
    #pragma omp simd
    for (i = 0; i < n_sys_; ++i) {
        double r = 1.0 / (1.0 - p.A1[i] * p.C0[i]);
        p.D0[i] =  r * (p.D0[i] - p.C0[i] * p.D1[i]);
        p.A0[i] =  r * p.A0[i];
        p.C0[i] =  0.0;
    }
    t1 = MPI_Wtime(); time_list[2] = t1 - t0;

    // 4) Single bidirectional exchange (merged 1-comm) — see solve_filtered_v1:
    //    my row N-1 (C,D) -> right ; my row 0 (A,D) -> left ; one round.
    MPI_Barrier(comm_); t0 = MPI_Wtime();
    {
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
    t1 = MPI_Wtime(); time_list[3] = t1 - t0;

    // 5) Solve both interface 2x2 blocks locally (no second round).
    MPI_Barrier(comm_); t0 = MPI_Wtime();
    {
        const bool has_left  = (left_rank_  != MPI_PROC_NULL);
        const bool has_right = (right_rank_ != MPI_PROC_NULL);
        if (has_left) {
            #pragma omp simd
            for (i = 0; i < n_sys_; ++i)
                p.D0[i] = (p.D0[i] - p.A0[i] * D_left[i]) / (1.0 - C_left[i] * p.A0[i]);
        }
        if (has_right) {
            #pragma omp simd
            for (i = 0; i < n_sys_; ++i)
                p.DN[i] = (p.DN[i] - p.CN[i] * D0_right[i]) / (1.0 - A_right[i] * p.CN[i]);
        }
    }
    t1 = MPI_Wtime(); time_list[4] = t1 - t0;
    time_list[5] = 0.0;

    // 8) Final local solve
    MPI_Barrier(comm_); t0 = MPI_Wtime();
    int J = cal_J_v1(p.D0, p.DN);
    const int j_left_end  = J;
    const int j_right_beg = (n_row_ - 1) - J;
    {
        double* Dj = D + (std::size_t)1 * n_sys_;
        double* Aj = A + (std::size_t)1 * n_sys_;
        for (j = 1; j <= j_left_end && j < n_row_ - 1; ++j, Dj += n_sys_, Aj += n_sys_) {
            #pragma omp simd
            for (i = 0; i < n_sys_; ++i) Dj[i] -= Aj[i] * p.D0[i];
        }
    }
    {
        int jrb = std::max(1, j_right_beg);
        double* Dj = D + (std::size_t)jrb * n_sys_;
        double* Cj = C + (std::size_t)jrb * n_sys_;
        for (j = jrb; j < n_row_ - 1; ++j, Dj += n_sys_, Cj += n_sys_) {
            #pragma omp simd
            for (i = 0; i < n_sys_; ++i) Dj[i] -= Cj[i] * p.DN[i];
        }
    }
    t1 = MPI_Wtime(); time_list[6] = t1 - t0;
}

// ============================================================================
//  solve_filtered_v2_profile() — per-phase timing with MPI_Barrier
// ============================================================================

void FilteredTDMA::solve_filtered_v2_profile(double* __restrict A,
                                             double* __restrict B,
                                             double* __restrict C,
                                             double* __restrict D,
                                             std::vector<double>& time_list) {
    time_list.resize(7);
    if (nprocs_ == 1) {
        double t0 = MPI_Wtime();
        tdma_many(A, B, C, D, n_sys_, n_row_);
        time_list.assign(7, 0.0);
        time_list[0] = MPI_Wtime() - t0;
        return;
    }

    int i, j;
    double t0, t1;
    const int J  = cal_J_rhs_bound(D);
    const int lo = (n_row_ - 2) - J;
    auto p = setup_ptrs(A, B, C, D, n_sys_, n_row_);

    const double* __restrict C_left   = C_left_recv_.data();
    const double* __restrict D_left   = D_left_recv_.data();
    const double* __restrict A_right  = A_right_recv_.data();
    const double* __restrict D0_right = D0_right_recv_.data();

    // 1) Forward Elimination
    MPI_Barrier(comm_); t0 = MPI_Wtime();
    #pragma omp simd
    for (i = 0; i < n_sys_; ++i) {
        double r0 = 1.0 / p.B0[i];
        p.A0[i] *= r0; p.C0[i] *= r0; p.D0[i] *= r0;
        double r1 = 1.0 / p.B1[i];
        p.A1[i] *= r1; p.C1[i] *= r1; p.D1[i] *= r1;
    }
    {
        double* Ajm = p.A1, *Cjm = p.C1, *Djm = p.D1;
        double* Aj = p.A1 + n_sys_, *Bj = p.B1 + n_sys_;
        double* Cj = p.C1 + n_sys_, *Dj = p.D1 + n_sys_;
        for (j = 2; j <= J; ++j,
             Ajm = Aj, Cjm = Cj, Djm = Dj,
             Aj += n_sys_, Bj += n_sys_, Cj += n_sys_, Dj += n_sys_) {
            #pragma omp simd
            for (i = 0; i < n_sys_; ++i) {
                double inv = 1.0 / (Bj[i] - Aj[i] * Cjm[i]);
                Dj[i] =  inv * (Dj[i] - Aj[i] * Djm[i]);
                Cj[i] =  inv * Cj[i];
                Aj[i] = -inv * Aj[i] * Ajm[i];
            }
        }
        for (j = J + 1; j < n_row_; ++j,
             Ajm = Aj, Cjm = Cj, Djm = Dj,
             Aj += n_sys_, Bj += n_sys_, Cj += n_sys_, Dj += n_sys_) {
            #pragma omp simd
            for (i = 0; i < n_sys_; ++i) {
                double inv = 1.0 / (Bj[i] - Aj[i] * Cjm[i]);
                Dj[i] = inv * (Dj[i] - Aj[i] * Djm[i]);
                Cj[i] = inv * Cj[i];
            }
        }
    }
    t1 = MPI_Wtime(); time_list[0] = t1 - t0;

    // 2) Backward Substitution
    MPI_Barrier(comm_); t0 = MPI_Wtime();
    {
        double* Cj = C + (std::size_t)(n_row_ - 3) * n_sys_;
        double* Dj = D + (std::size_t)(n_row_ - 3) * n_sys_;
        double* Djp = Dj + n_sys_;
        for (j = n_row_ - 3; j >= 1; --j) {
            #pragma omp simd
            for (i = 0; i < n_sys_; ++i) Dj[i] -= Cj[i] * Djp[i];
            Djp = Dj; Dj -= n_sys_; Cj -= n_sys_;
        }
    }
    if (J >= 2) {
        double* Aj = A + (std::size_t)(J - 1) * n_sys_;
        double* Cj = C + (std::size_t)(J - 1) * n_sys_;
        double* Ajp = Aj + n_sys_;
        for (j = J - 1; j >= 1; --j) {
            #pragma omp simd
            for (i = 0; i < n_sys_; ++i) Aj[i] -= Cj[i] * Ajp[i];
            Ajp = Aj; Aj -= n_sys_; Cj -= n_sys_;
        }
    }
    {
        double* Cjp = C + (std::size_t)(n_row_ - 2) * n_sys_;
        double* Cj  = Cjp - n_sys_;
        for (j = n_row_ - 3; j >= lo + 1; --j) {
            #pragma omp simd
            for (i = 0; i < n_sys_; ++i) Cj[i] *= -Cjp[i];
            Cjp = Cj; Cj -= n_sys_;
        }
    }
    t1 = MPI_Wtime(); time_list[1] = t1 - t0;

    // 3) Pack — C0 = 0
    MPI_Barrier(comm_); t0 = MPI_Wtime();
    #pragma omp simd
    for (i = 0; i < n_sys_; ++i) {
        double r = 1.0 / (1.0 - p.A1[i] * p.C0[i]);
        p.D0[i] =  r * (p.D0[i] - p.C0[i] * p.D1[i]);
        p.A0[i] =  r * p.A0[i];
    }
    t1 = MPI_Wtime(); time_list[2] = t1 - t0;

    // 4) Single bidirectional exchange (merged 1-comm) — see solve_filtered_v1:
    //    my row N-1 (C,D) -> right ; my row 0 (A,D) -> left ; one round.
    MPI_Barrier(comm_); t0 = MPI_Wtime();
    {
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
    t1 = MPI_Wtime(); time_list[3] = t1 - t0;

    // 5) Solve both interface 2x2 blocks locally (no second round).
    MPI_Barrier(comm_); t0 = MPI_Wtime();
    {
        const bool has_left  = (left_rank_  != MPI_PROC_NULL);
        const bool has_right = (right_rank_ != MPI_PROC_NULL);
        if (has_left) {
            #pragma omp simd
            for (i = 0; i < n_sys_; ++i)
                p.D0[i] = (p.D0[i] - p.A0[i] * D_left[i]) / (1.0 - C_left[i] * p.A0[i]);
        }
        if (has_right) {
            #pragma omp simd
            for (i = 0; i < n_sys_; ++i)
                p.DN[i] = (p.DN[i] - p.CN[i] * D0_right[i]) / (1.0 - A_right[i] * p.CN[i]);
        }
    }
    t1 = MPI_Wtime(); time_list[4] = t1 - t0;
    time_list[5] = 0.0;

    // 8) Final local solve
    MPI_Barrier(comm_); t0 = MPI_Wtime();
    const int j_left_end  = J;
    const int j_right_beg = lo + 1;
    {
        double* Dj = D + (std::size_t)1 * n_sys_;
        double* Aj = A + (std::size_t)1 * n_sys_;
        for (j = 1; j <= j_left_end; ++j, Dj += n_sys_, Aj += n_sys_) {
            #pragma omp simd
            for (i = 0; i < n_sys_; ++i) Dj[i] -= Aj[i] * p.D0[i];
        }
    }
    {
        double* Dj = D + (std::size_t)j_right_beg * n_sys_;
        double* Cj = C + (std::size_t)j_right_beg * n_sys_;
        for (j = j_right_beg; j < n_row_ - 1; ++j, Dj += n_sys_, Cj += n_sys_) {
            #pragma omp simd
            for (i = 0; i < n_sys_; ++i) Dj[i] -= Cj[i] * p.DN[i];
        }
    }
    t1 = MPI_Wtime(); time_list[6] = t1 - t0;
}