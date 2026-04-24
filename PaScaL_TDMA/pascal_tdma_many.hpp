#ifndef PASCAL_TDMA_MANY_HPP
#define PASCAL_TDMA_MANY_HPP

#include <mpi.h>
#include <vector>

/// Parallel tridiagonal solver for multiple RHS using the PaScaL algorithm.
///
/// Encapsulates the MPI communication plan (alltoall transpose) and provides:
///   - solve()         : standard solve
///   - solve_profile() : same solve with per-phase timing (7 entries)
///
/// Memory layout: row-major [n_row × n_sys], i.e. row j starts at offset j*n_sys.
class PaScaLTDMAMany {
public:
    /// Construct solver for `n_sys` independent systems.
    /// `n_sys` is the number of parallel tridiagonal systems.
    PaScaLTDMAMany(int n_sys, int myrank, int nprocs, MPI_Comm comm);

    ~PaScaLTDMAMany();

    // Non-copyable
    PaScaLTDMAMany(const PaScaLTDMAMany&)            = delete;
    PaScaLTDMAMany& operator=(const PaScaLTDMAMany&) = delete;

    /// Solve n_sys tridiagonal systems each of size n_row.
    /// A, B, C: lower/diagonal/upper coefficients [n_row × n_sys].
    /// D: right-hand side [n_row × n_sys]; overwritten with solution on return.
    void solve(double* __restrict A, double* __restrict B,
               double* __restrict C, double* __restrict D,
               int n_sys, int n_row);

    /// Same as solve() but for periodic (cyclic) tridiagonal systems.
    /// Row 0 connects to row n_row-1 via A[0] and C[n_row-1].
    void solve_cyclic(double* __restrict A, double* __restrict B,
                      double* __restrict C, double* __restrict D,
                      int n_sys, int n_row);

    // --- Profile: per-phase timing with MPI_Barrier (7 entries) ---
    void solve_profile(double* __restrict A, double* __restrict B,
                       double* __restrict C, double* __restrict D,
                       int n_sys, int n_row, std::vector<double>& time_list);

    int n_sys_rt() const { return n_sys_rt_; }
    int n_row_rt() const { return n_row_rt_; }
    int nprocs()   const { return nprocs_; }

private:
    MPI_Comm comm_;
    int      nprocs_;
    int      n_sys_rt_;   // systems assigned to this rank after transpose
    int      n_row_rt_;   // rows of reduced system after transpose (= 2 * nprocs)

    // Derived datatypes for alltoall: send [n_row_rd × n_sys] subarray slices
    std::vector<MPI_Datatype> ddtype_Fs_;
    std::vector<MPI_Datatype> ddtype_Bs_;

    // Alltoall counts/displacements (all 1 / 0 because DDT handles layout)
    std::vector<int> count_send_, displ_send_;
    std::vector<int> count_recv_, displ_recv_;

    // Reduced system buffers [2 × n_sys]: boundary rows from each rank
    std::vector<double> A_rd_, B_rd_, C_rd_, D_rd_;

    // Transposed system buffers [n_row_rt × n_sys_rt]: local reduced block
    std::vector<double> A_rt_, B_rt_, C_rt_, D_rt_;
};

#endif // PASCAL_TDMA_MANY_HPP
