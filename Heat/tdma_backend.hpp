#ifndef HEAT_TDMA_BACKEND_HPP
#define HEAT_TDMA_BACKEND_HPP

// Heat-local wrapper that lets SolveTheta pick between Filtered_TDMA and
// PaScaL_TDMA at runtime via the `tdma_backend` input parameter.
//
// Layout matches the existing SolveTheta usage: row-major [n_row x n_sys].
// For the Filtered backend the per-row spectral bounds A_rho()/C_rho() must
// be filled by the caller before solve(); for the PaScaL backend that step
// is a no-op (the corresponding methods return a writable scratch buffer so
// existing call sites keep compiling).

#include <memory>
#include <mpi.h>
#include <string>
#include <vector>

#include "filtered_tdma.hpp"
#include "pascal_tdma_many.hpp"

class TdmaBackend {
public:
    enum class Kind { FILTERED, PASCAL };

    // "filtered" (default) or "pascal" (case-insensitive); also accepts
    // "pascal_tdma" / "filtered_tdma".
    static Kind parse(const std::string& s);

    TdmaBackend(Kind kind, int n_sys, int n_row,
                int myrank, int nprocs, MPI_Comm comm,
                int left_rank, int right_rank,
                double eps_constant);

    void solve(double* A, double* B, double* C, double* D);
    void solve_profile(double* A, double* B, double* C, double* D,
                       std::vector<double>& time_list);

    // FILTERED: refresh per-row spectral bounds A_rho/C_rho from current
    //           matrix entries (mirrors channel/TdmaSolver::set_rho).
    // PASCAL  : no-op.
    void set_rho(const double* A, const double* B, const double* C);

    // FILTERED: forwards to FilteredTDMA::set_eps_constant (channel passes dt).
    // PASCAL  : no-op.
    void set_eps_constant(double eps_constant);

    // For FILTERED: alias of the underlying FilteredTDMA::A_rho() / C_rho().
    // For PASCAL  : returns a scratch vector of size n_row+1 so existing
    //               write-through call sites compile and are harmless.
    std::vector<double>& A_rho();
    std::vector<double>& C_rho();

    Kind kind() const { return kind_; }

private:
    Kind kind_;
    int  n_sys_, n_row_;
    std::unique_ptr<FilteredTDMA>   filt_;
    std::unique_ptr<PaScaLTDMAMany> pasc_;
    std::vector<double> dummy_rho_;   // PASCAL only
};

#endif // HEAT_TDMA_BACKEND_HPP
