#include "filtered_tdma.hpp"
#include "tdma_local.hpp"

#include <cstddef>
#include <cmath>
#include <algorithm>
#include <vector>

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
//  Helper: common pointer setup for all solve variants
// ============================================================================

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
//  solve_filtered_v1() — no timing
// ============================================================================

void FilteredTDMA::solve_filtered_v1(double* __restrict A,
                                     double* __restrict B,
                                     double* __restrict C,
                                     double* __restrict D) {
    if (nprocs_ == 1) {
        tdma_many(A, B, C, D, n_sys_, n_row_);
        return;
    }

    int i, j;
    auto p = setup_ptrs(A, B, C, D, n_sys_, n_row_);

    const double* __restrict C_left   = C_left_recv_.data();
    const double* __restrict D_left   = D_left_recv_.data();
    const double* __restrict A_right  = A_right_recv_.data();
    const double* __restrict D0_right = D0_right_recv_.data();

    // 1) Forward Elimination
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

    // 2) Backward Substitution
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

    // 3) Pack reduced system — C0 = 0 to decouple row 0 from x_{N-1}
    #pragma omp simd
    for (i = 0; i < n_sys_; ++i) {
        double r = 1.0 / (1.0 - p.A1[i] * p.C0[i]);
        p.D0[i] =  r * (p.D0[i] - p.C0[i] * p.D1[i]);
        p.A0[i] =  r * p.A0[i];
    }

    // 4) Single bidirectional exchange of both interfaces' reduced rows:
    //      my row N-1 (C,D) -> right   ;  my row 0 (A,D) -> left
    //    recv: left's row N-1 (LEFT 2x2), right's row 0 (RIGHT 2x2).
    //    Replaces the old 2-round forward/back-communication: the reduced
    //    system decouples into independent interface 2x2 blocks, so each rank
    //    solves BOTH its interfaces locally (redundant 2x2, one comm round).
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

    // 5) Solve both interface 2x2 blocks locally (no second round):
    //      LEFT  : x_0   = (D0 - A0*D_left)  / (1 - C_left*A0)
    //      RIGHT : x_N-1 = (DN - CN*D0_right)/ (1 - A_right*CN)
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

    // 8) Final local solve (boundary correction with J cutoff)
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
}

// ============================================================================
//  solve_filtered_v2() — no timing
// ============================================================================

void FilteredTDMA::solve_filtered_v2(double* __restrict A,
                                     double* __restrict B,
                                     double* __restrict C,
                                     double* __restrict D) {
    if (nprocs_ == 1) {
        tdma_many(A, B, C, D, n_sys_, n_row_);
        return;
    }

    int i, j;
    auto p = setup_ptrs(A, B, C, D, n_sys_, n_row_);

    const int J  = cal_J_rhs_bound(D);
    const int lo = (n_row_ - 2) - J;

    const double* __restrict C_left   = C_left_recv_.data();
    const double* __restrict D_left   = D_left_recv_.data();
    const double* __restrict A_right  = A_right_recv_.data();
    const double* __restrict D0_right = D0_right_recv_.data();

    // 1) Forward Elimination
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

        // Phase 1: j=2..J — full update (D, C, A)
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
        // Phase 2: j=J+1..n_row-1 — skip A update
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

    // 2) Backward Substitution — 3 separate phases
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

    // 3) Pack — C0 = 0
    #pragma omp simd
    for (i = 0; i < n_sys_; ++i) {
        double r = 1.0 / (1.0 - p.A1[i] * p.C0[i]);
        p.D0[i] =  r * (p.D0[i] - p.C0[i] * p.D1[i]);
        p.A0[i] =  r * p.A0[i];
    }

    // 4) Single bidirectional exchange (see solve_filtered_v1 for derivation):
    //      my row N-1 (C,D) -> right ;  my row 0 (A,D) -> left
    //      recv left's row N-1 (LEFT 2x2) and right's row 0 (RIGHT 2x2).
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

    // 5) Solve both interface 2x2 blocks locally (no second round).
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

    // 8) Final local solve
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
}