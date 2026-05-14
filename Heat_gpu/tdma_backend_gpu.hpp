#ifndef HEAT_GPU_TDMA_BACKEND_GPU_HPP
#define HEAT_GPU_TDMA_BACKEND_GPU_HPP

// Runtime backend selector for the Heat_gpu solver.
//
//   parse("filtered") → FILTERED → FilteredTDMACUDA
//   parse("pascal")   → PASCAL   → PaScaLTDMAManyCUDA
//
// Both backends operate on device pointers; the CPU `TdmaBackend` (Heat/) is
// the host-side analogue.  Only the FILTERED backend has spectral-radius and
// epsilon book-keeping; the PASCAL methods are no-ops.

#include <memory>
#include <mpi.h>
#include <string>
#include <vector>

#include "filtered_tdma_cuda.hpp"
#include "pascal_tdma_many_cuda.hpp"

class TdmaBackendGPU {
public:
    enum class Kind { FILTERED, FILTERED_V2, PASCAL };

    /// Case-insensitive. Accepted strings:
    ///   "pascal" | "pascal_tdma"                  → PASCAL
    ///   "filtered" | "filtered_tdma" | "filtered_v1" → FILTERED   (solve_filtered_v1)
    ///   "filtered_v2"                              → FILTERED_V2 (solve_filtered_v2)
    static Kind parse(const std::string& s);

    /// `n_sys`, `n_row` are the dimensions of one direction's system.
    /// (FILTERED uses both; PASCAL needs only n_sys.)
    TdmaBackendGPU(Kind kind, int n_sys, int n_row,
                   int myrank, int nprocs, MPI_Comm comm,
                   int left_rank, int right_rank,
                   double eps_constant);

    /// Solve all n_sys tridiagonal systems of length n_row.
    /// FILTERED: solve_filtered_v1; PASCAL: PaScaLTDMAManyCUDA::solve.
    void solve(double* d_A, double* d_B, double* d_C, double* d_D);

    // FILTERED only — refresh per-row spectral bounds. No-op for PASCAL.
    void set_rho_device(const double* d_A, const double* d_B, const double* d_C);

    // FILTERED only — refresh eps from a new eps_constant. No-op for PASCAL.
    void set_eps_constant(double eps_constant);

    Kind kind() const { return kind_; }

private:
    Kind kind_;
    int  n_sys_, n_row_;
    std::unique_ptr<FilteredTDMACUDA>   filt_;
    std::unique_ptr<PaScaLTDMAManyCUDA> pasc_;
};

#endif // HEAT_GPU_TDMA_BACKEND_GPU_HPP
