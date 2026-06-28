#include "TdmaSolverGPU.hpp"

#include "MpiTopology.hpp"

#include <algorithm>
#include <cctype>

namespace channel {

TdmaSolverGPU::Backend TdmaSolverGPU::parse_backend(const std::string& s)
{
    std::string t;
    t.reserve(s.size());
    for (char c : s) t.push_back(static_cast<char>(std::tolower(c)));
    if (t == "pascal" || t == "pascal_tdma") return Backend::PASCAL;
    return Backend::FILTERED;
}

TdmaSolverGPU::TdmaSolverGPU(const MpiTopology& topo, int axis,
                             int n_sys, int n_row, bool periodic,
                             Backend backend, double eps_constant)
    : backend_(backend), n_sys_(n_sys), n_row_(n_row), periodic_(periodic)
{
    const int rank = topo.rank_in(axis);
    const int nprocs = topo.size_in(axis);
    MPI_Comm comm = topo.comm(axis);

    if (backend_ == Backend::FILTERED) {
        filt_ = std::make_unique<FilteredTDMACUDA>(
            n_sys, n_row, rank, nprocs, comm,
            topo.left_in_sub(axis), topo.right_in_sub(axis), eps_constant);
    } else {
        pasc_ = std::make_unique<PaScaLTDMACUDA>(n_sys, rank, nprocs, comm, 128, 1);
    }
}

void TdmaSolverGPU::solve(double* d_A, double* d_B, double* d_C, double* d_D)
{
    if (backend_ == Backend::FILTERED)
        filt_->solve_filtered_v2(d_A, d_B, d_C, d_D);
    else
        pasc_->solve(d_A, d_B, d_C, d_D, n_sys_, n_row_);
}

void TdmaSolverGPU::solve_cycl(double* d_A, double* d_B, double* d_C, double* d_D)
{
    if (backend_ == Backend::FILTERED)
        filt_->solve_cycl_filtered_v2(d_A, d_B, d_C, d_D);
    else
        pasc_->solve_cyclic(d_A, d_B, d_C, d_D, n_sys_, n_row_);
}

void TdmaSolverGPU::set_rho_device(const double* d_A, const double* d_B,
                                   const double* d_C)
{
    if (backend_ == Backend::FILTERED) filt_->set_rho_device(d_A, d_B, d_C);
}

void TdmaSolverGPU::set_eps_constant(double eps_constant)
{
    if (backend_ == Backend::FILTERED) filt_->set_eps_constant(eps_constant);
}

double TdmaSolverGPU::last_comm_ms() const
{
    return backend_ == Backend::FILTERED ? filt_->last_comm_ms() : pasc_->last_comm_ms();
}

double TdmaSolverGPU::last_gpu_ms() const
{
    return backend_ == Backend::FILTERED ? filt_->last_gpu_ms() : pasc_->last_gpu_ms();
}

} // namespace channel
