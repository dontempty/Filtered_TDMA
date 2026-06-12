// channel/TdmaSolver.hpp
//
// Unified tridiagonal-solver wrapper for the Beam-Warming ADI momentum sweep.
// Runtime backend selection via the `Backend` enum:
//   FILTERED  — Filtered_TDMA library (truncated-filter v2)
//   PASCAL    — PaScaL_TDMA library (reduced parallel Thomas)
//
// Both backends use the same row-major memory layout [n_row × n_sys].
// `set_rho()` is only meaningful for FILTERED (spectral-radius cutoff);
// it is a no-op for PASCAL.
//
// The solver for one MomentumSolver axis is constructed once and reused
// across steps. Coefficients and RHS are written by the caller into
// pre-allocated arrays, then passed to solve() or solve_cycl().

#ifndef CHANNEL_TDMA_SOLVER_HPP
#define CHANNEL_TDMA_SOLVER_HPP

#include <memory>
#include <mpi.h>
#include <string>

#include "filtered_tdma.hpp"
#include "pascal_tdma_many.hpp"

namespace channel {

class MpiTopology;

class TdmaSolver {
public:
    enum class Backend { FILTERED, PASCAL };

    /// Accepts "filtered" or "pascal" (case-insensitive); defaults to FILTERED.
    static Backend parse_backend(const std::string& s);

    TdmaSolver(const MpiTopology& topo, int axis,
               int n_sys, int n_row, bool periodic,
               Backend backend, double eps_constant = 1.0e-12);

    void solve     (double* A, double* B, double* C, double* D);
    void solve_cycl(double* A, double* B, double* C, double* D);

    /// FILTERED non-cyclic + sub-phase timing. Fills time_list[0..6] with:
    ///   [0]=fwd elim  [1]=bwd sub  [2]=pack  [3]=MPI4  [4]=unpack+MPI2
    ///   [5]=unused    [6]=cal_J + final correction
    /// PASCAL: time_list[0..5]=0, time_list[6]=total wall time.
    /// solve_cycl path: caller should time the whole solve_cycl() call as one
    /// chunk; this method only handles non-cyclic.
    void solve_profile(double* A, double* B, double* C, double* D,
                       std::vector<double>& time_list);

    /// FILTERED: updates truncation radius from first-column |A/B|, |C/B|.
    /// PASCAL   : no-op.
    void set_rho(const double* A, const double* B, const double* C, int n_sys);

    /// FILTERED: forwards to underlying FilteredTDMA::set_eps_constant.
    /// PASCAL   : no-op.
    void set_eps_constant(double eps_constant);

    int     n_sys()    const { return n_sys_; }
    int     n_row()    const { return n_row_; }
    bool    periodic() const { return periodic_; }
    Backend backend()  const { return backend_; }

private:
    Backend backend_;
    int     n_sys_   = 0;
    int     n_row_   = 0;
    bool    periodic_ = false;

    // Only one is allocated based on backend_
    std::unique_ptr<FilteredTDMA>   filt_;   // FILTERED
    std::unique_ptr<PaScaLTDMAMany> pasc_;   // PASCAL
};

} // namespace channel

#endif // CHANNEL_TDMA_SOLVER_HPP
