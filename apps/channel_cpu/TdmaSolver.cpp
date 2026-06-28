#include "TdmaSolver.hpp"

#include "MpiTopology.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>

namespace channel {

TdmaSolver::Backend TdmaSolver::parse_backend(const std::string& s)
{
    std::string t;
    t.reserve(s.size());
    for (char c : s) t.push_back(static_cast<char>(std::tolower(c)));
    if (t == "pascal" || t == "pascal_tdma") return Backend::PASCAL;
    return Backend::FILTERED;   // default
}

TdmaSolver::TdmaSolver(const MpiTopology& topo, int axis,
                       int n_sys, int n_row, bool periodic,
                       Backend backend, double eps_constant)
    : backend_(backend), n_sys_(n_sys), n_row_(n_row), periodic_(periodic)
{
    const int rank   = topo.rank_in(axis);
    const int nprocs = topo.size_in(axis);
    MPI_Comm  comm   = topo.comm(axis);

    if (backend_ == Backend::FILTERED) {
        const int left  = topo.left_in_sub(axis);
        const int right = topo.right_in_sub(axis);
        filt_ = std::make_unique<FilteredTDMA>(n_sys, n_row,
                                               rank, nprocs, comm,
                                               left, right, eps_constant);
    } else {
        pasc_ = std::make_unique<PaScaLTDMA>(n_sys, rank, nprocs, comm);
    }
}

void TdmaSolver::set_rho(const double* A, const double* B,
                         const double* C, int n_sys)
{
    if (backend_ != Backend::FILTERED) return;   // PaScaL: no-op

    auto& ar = filt_->A_rho();
    auto& cr = filt_->C_rho();
    for (int k = 0; k < n_row_; ++k) {
        double bk = B[k * n_sys];
        ar[k] = (bk != 0.0) ? std::abs(A[k * n_sys] / bk) : 0.0;
        cr[k] = (bk != 0.0) ? std::abs(C[k * n_sys] / bk) : 0.0;
    }
    ar[n_row_] = 0.0;
    cr[n_row_] = 0.0;
}

void TdmaSolver::set_eps_constant(double eps_constant)
{
    if (backend_ == Backend::FILTERED) filt_->set_eps_constant(eps_constant);
    // PaScaL: no-op
}

void TdmaSolver::solve(double* A, double* B, double* C, double* D)
{
    if (backend_ == Backend::FILTERED) {
        filt_->solve_filtered_v2(A, B, C, D);
    } else {
        pasc_->solve(A, B, C, D, n_sys_, n_row_);
    }
}

void TdmaSolver::solve_cycl(double* A, double* B, double* C, double* D)
{
    if (backend_ == Backend::FILTERED) {
        filt_->solve_cycl_filtered_v2(A, B, C, D);
    } else {
        pasc_->solve_cyclic(A, B, C, D, n_sys_, n_row_);
    }
}

} // namespace channel
