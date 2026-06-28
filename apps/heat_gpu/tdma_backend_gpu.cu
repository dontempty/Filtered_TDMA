#include "tdma_backend_gpu.hpp"

#include <algorithm>
#include <cctype>

TdmaBackendGPU::Kind TdmaBackendGPU::parse(const std::string& s) {
    std::string t;
    t.reserve(s.size());
    for (char c : s) t.push_back(static_cast<char>(std::tolower(c)));
    if (t == "pascal" || t == "pascal_tdma") return Kind::PASCAL;
    if (t == "filtered_v2")                  return Kind::FILTERED_V2;
    return Kind::FILTERED;  // "filtered", "filtered_v1", "filtered_tdma"
}

TdmaBackendGPU::TdmaBackendGPU(Kind kind, int n_sys, int n_row,
                               int myrank, int nprocs, MPI_Comm comm,
                               int left_rank, int right_rank,
                               double eps_constant)
    : kind_(kind), n_sys_(n_sys), n_row_(n_row)
{
    if (kind_ == Kind::PASCAL) {
        // (128, 1): warp-aligned, full coalescing.  Replaces legacy (16, 16)
        // which broke warp memory coalescing on modified_thomas / tdma_many.
        // See REPORT_OPTIMIZATION_SUMMARY — modT -33%, total solve -16~20%.
        pasc_ = std::make_unique<PaScaLTDMAManyCUDA>(n_sys, myrank, nprocs, comm,
                                                    128, 1);
    } else {
        filt_ = std::make_unique<FilteredTDMACUDA>(n_sys, n_row,
                                                   myrank, nprocs, comm,
                                                   left_rank, right_rank,
                                                   eps_constant);
    }
}

void TdmaBackendGPU::solve(double* d_A, double* d_B, double* d_C, double* d_D) {
    // heat_gpu is a benchmark: route all backends through their *_profile variants
    // so last_comm_ms()/last_gpu_ms() timers are populated for CSV export.
    // Numerics are identical; profiling adds cudaStreamSynchronize + cudaEvent brackets.
    if (kind_ == Kind::FILTERED) {
        filt_->solve_filtered_v1_profile(d_A, d_B, d_C, d_D);
    } else if (kind_ == Kind::FILTERED_V2) {
        filt_->solve_filtered_v2_profile(d_A, d_B, d_C, d_D);
    } else {
        pasc_->solve_profile(d_A, d_B, d_C, d_D, n_sys_, n_row_);
    }
}

void TdmaBackendGPU::solve_cyclic(double* d_A, double* d_B, double* d_C, double* d_D) {
    if (kind_ == Kind::FILTERED) {
        filt_->solve_cycl_filtered_v1_profile(d_A, d_B, d_C, d_D);
    } else if (kind_ == Kind::FILTERED_V2) {
        filt_->solve_cycl_filtered_v2_profile(d_A, d_B, d_C, d_D);
    } else {
        pasc_->solve_cyclic_profile(d_A, d_B, d_C, d_D, n_sys_, n_row_);
    }
}

void TdmaBackendGPU::set_rho_device(const double* d_A,
                                    const double* d_B,
                                    const double* d_C) {
    if (kind_ != Kind::PASCAL) filt_->set_rho_device(d_A, d_B, d_C);
}

void TdmaBackendGPU::set_eps_constant(double eps_constant) {
    if (kind_ != Kind::PASCAL) filt_->set_eps_constant(eps_constant);
}
