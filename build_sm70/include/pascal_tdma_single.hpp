#ifndef PASCAL_TDMA_SINGLE_HPP
#define PASCAL_TDMA_SINGLE_HPP

#include <mpi.h>
#include <vector>

/// Parallel tridiagonal solver for a single RHS using MPI_Gather/Scatter.
///
/// Reduces the local system to a 2-row boundary system, gathers all boundary
/// rows to `gather_rank`, solves the reduced system there, then scatters
/// back and completes the local back-substitution.
///
/// Provides:
///   - solve()        : standard tridiagonal solve
///   - solve_cyclic() : cyclic tridiagonal solve
class PaScaLTDMASingle {
public:
    /// Construct solver.
    /// `gather_rank` is the rank that collects and solves the reduced system.
    PaScaLTDMASingle(int myrank, int nprocs, MPI_Comm comm, int gather_rank);

    ~PaScaLTDMASingle() = default;

    // Non-copyable
    PaScaLTDMASingle(const PaScaLTDMASingle&)            = delete;
    PaScaLTDMASingle& operator=(const PaScaLTDMASingle&) = delete;

    /// Solve a single tridiagonal system of size n_row.
    /// a, b, c: lower/diagonal/upper coefficients (size n_row).
    /// d: right-hand side (size n_row); overwritten with solution on return.
    void solve(std::vector<double>& a, std::vector<double>& b,
               std::vector<double>& c, std::vector<double>& d, int n_row);

    /// Cyclic variant: handles periodic boundary conditions.
    void solve_cyclic(std::vector<double>& a, std::vector<double>& b,
                      std::vector<double>& c, std::vector<double>& d, int n_row);

    int myrank()     const { return myrank_; }
    int nprocs()     const { return nprocs_; }
    int gather_rank() const { return gather_rank_; }

private:
    MPI_Comm comm_;
    int      myrank_, nprocs_, gather_rank_;
    int      n_row_rt_;   // rows of gathered reduced system (= 2 * nprocs)

    // Reduced system per rank (2 rows: first and last)
    std::vector<double> A_rd_, B_rd_, C_rd_, D_rd_;

    // Gathered reduced system on gather_rank (n_row_rt rows)
    std::vector<double> A_rt_, B_rt_, C_rt_, D_rt_;
};

#endif // PASCAL_TDMA_SINGLE_HPP
