#ifndef FILTERED_TDMA_HPP
#define FILTERED_TDMA_HPP

#include <mpi.h>
#include <vector>

/// Parallel tridiagonal solver for multiple RHS using the DistD2 algorithm.
///
/// Encapsulates the MPI communication plan and provides two solve variants:
///   - solve_filtered_v1() : skips only "Final local solve"
///   - solve_filtered_v2() : skips "Forward", "Backward" and "Final local solve"
///
/// Both methods accept an optional time_list pointer to collect per-phase timing.
class FilteredTDMA {
public:
    /// Construct solver for `n_sys` independent systems, each of size `n_row`.
    FilteredTDMA(int n_sys, int n_row,
                 int myrank, int nprocs, MPI_Comm comm,
                 int left_rank, int right_rank,
                 double eps_constant);

    ~FilteredTDMA();

    // Non-copyable
    FilteredTDMA(const FilteredTDMA&)            = delete;
    FilteredTDMA& operator=(const FilteredTDMA&) = delete;

    /// Filtered_TDMA (v1).
    void solve_filtered_v1(double* __restrict A, double* __restrict B,
                           double* __restrict C, double* __restrict D);

    void solve_cycl_filtered_v1(double* __restrict A, double* __restrict B,
                                double* __restrict C, double* __restrict D);

    /// Filtered_TDMA (v2).
    void solve_filtered_v2(double* __restrict A, double* __restrict B,
                           double* __restrict C, double* __restrict D);

    void solve_cycl_filtered_v2(double* __restrict A, double* __restrict B,
                                double* __restrict C, double* __restrict D);

    // --- Profile: per-phase timing with MPI_Barrier (7 entries) ---
    void solve_filtered_v1_profile(double* __restrict A, double* __restrict B,
                                   double* __restrict C, double* __restrict D,
                                   std::vector<double>& time_list);
    void solve_filtered_v2_profile(double* __restrict A, double* __restrict B,
                                   double* __restrict C, double* __restrict D,
                                   std::vector<double>& time_list);

    // --- Accessors for external coefficient setup (used by ADI driver) ---
    std::vector<double>& A_rho() { return A_rho_; }
    std::vector<double>& C_rho() { return C_rho_; }

    int n_sys()  const { return n_sys_; }
    int n_row()  const { return n_row_; }
    int nprocs() const { return nprocs_; }

    /// Update eps from a new eps_constant (e.g. the current dt).
    /// eps_ = eps_constant / (n_row * nprocs)^2
    void set_eps_constant(double eps_constant) {
        double N = static_cast<double>(n_row_ * nprocs_);
        eps_ = eps_constant / (N * N);
    }

private:
    /// Estimate cutoff index J using ||A^{-1}|| * ||b|| bound:
    ///   B = q(2+q) / ((1-q)(1-2rho)) * max|D_rhs|
    /// D must point to the full [n_row x n_sys] RHS array before elimination.
    int cal_J_rhs_bound(const double* D);
    /// Legacy estimators kept for reference (no longer called).
    int cal_J_v1(const double* D0, const double* DN);
    int cal_J_v2(double D0, double DN);

    // --- MPI plan data ---
    MPI_Comm     comm_;
    int          n_sys_, n_row_, nprocs_;
    int          left_rank_, right_rank_;

    MPI_Datatype row0_type_;      // subarray type for row j=0
    MPI_Datatype rowN_type_;      // subarray type for row j=n_row-1

    // Receive buffers for neighbor boundary data
    std::vector<double> C_left_recv_;
    std::vector<double> D_left_recv_;

    std::vector<double> D_right_send_;
    std::vector<double> D_right_recv_;

    // Spectral radius estimates (set externally before each solve)
    std::vector<double> A_rho_;
    std::vector<double> C_rho_;

    double eps_;
};

#endif // FILTERED_TDMA_HPP
