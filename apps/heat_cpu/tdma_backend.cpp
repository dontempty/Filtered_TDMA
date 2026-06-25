#include "tdma_backend.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>

TdmaBackend::Kind TdmaBackend::parse(const std::string& s) {
    std::string t;
    t.reserve(s.size());
    for (char c : s) t.push_back(static_cast<char>(std::tolower(c)));
    if (t == "pascal" || t == "pascal_tdma") return Kind::PASCAL;
    if (t == "filtered_v2")                  return Kind::FILTERED_V2;
    return Kind::FILTERED;
}

TdmaBackend::TdmaBackend(Kind kind, int n_sys, int n_row,
                         int myrank, int nprocs, MPI_Comm comm,
                         int left_rank, int right_rank,
                         double eps_constant)
    : kind_(kind), n_sys_(n_sys), n_row_(n_row)
{
    if (kind_ == Kind::FILTERED || kind_ == Kind::FILTERED_V2) {
        filt_ = std::make_unique<FilteredTDMA>(n_sys, n_row,
                                               myrank, nprocs, comm,
                                               left_rank, right_rank,
                                               eps_constant);
    } else {
        pasc_ = std::make_unique<PaScaLTDMAMany>(n_sys, myrank, nprocs, comm);
        dummy_rho_.assign(n_row + 1, 0.0);
    }
}

void TdmaBackend::set_rho(const double* A, const double* B, const double* C) {
    if (kind_ != Kind::FILTERED && kind_ != Kind::FILTERED_V2) return;
    auto& ar = filt_->A_rho();
    auto& cr = filt_->C_rho();
    for (int k = 0; k < n_row_; ++k) {
        const double bk = B[k * n_sys_];
        ar[k] = (bk != 0.0) ? std::abs(A[k * n_sys_] / bk) : 0.0;
        cr[k] = (bk != 0.0) ? std::abs(C[k * n_sys_] / bk) : 0.0;
    }
    ar[n_row_] = 0.0;
    cr[n_row_] = 0.0;
}

void TdmaBackend::set_eps_constant(double eps_constant) {
    if (kind_ == Kind::FILTERED || kind_ == Kind::FILTERED_V2)
        filt_->set_eps_constant(eps_constant);
}

void TdmaBackend::solve(double* A, double* B, double* C, double* D) {
    if (kind_ == Kind::FILTERED) {
        filt_->solve_filtered_v1(A, B, C, D);
    } else if (kind_ == Kind::FILTERED_V2) {
        filt_->solve_filtered_v2(A, B, C, D);
    } else {
        pasc_->solve(A, B, C, D, n_sys_, n_row_);
    }
}

void TdmaBackend::solve_cyclic(double* A, double* B, double* C, double* D) {
    if (kind_ == Kind::FILTERED) {
        filt_->solve_cycl_filtered_v1(A, B, C, D);
    } else if (kind_ == Kind::FILTERED_V2) {
        filt_->solve_cycl_filtered_v2(A, B, C, D);
    } else {
        pasc_->solve_cyclic(A, B, C, D, n_sys_, n_row_);
    }
}

void TdmaBackend::solve_profile(double* A, double* B, double* C, double* D,
                                std::vector<double>& time_list) {
    if (kind_ == Kind::FILTERED) {
        filt_->solve_filtered_v1_profile(A, B, C, D, time_list);
    } else if (kind_ == Kind::FILTERED_V2) {
        filt_->solve_filtered_v2_profile(A, B, C, D, time_list);
    } else {
        pasc_->solve_profile(A, B, C, D, n_sys_, n_row_, time_list);
    }
}

std::vector<double>& TdmaBackend::A_rho() {
    return (filt_) ? filt_->A_rho() : dummy_rho_;
}

std::vector<double>& TdmaBackend::C_rho() {
    return (filt_) ? filt_->C_rho() : dummy_rho_;
}
