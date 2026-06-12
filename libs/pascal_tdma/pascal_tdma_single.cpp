#include "pascal_tdma_single.hpp"
#include "tdma_local.hpp"

#include <vector>

// ============================================================================
//  Constructor
// ============================================================================

PaScaLTDMASingle::PaScaLTDMASingle(int myrank, int nprocs, MPI_Comm comm, int gather_rank)
    : comm_(comm), myrank_(myrank), nprocs_(nprocs), gather_rank_(gather_rank)
{
    // Each rank contributes 2 boundary rows to the reduced system
    const int n_row_rd = 2;
    n_row_rt_ = n_row_rd * nprocs;

    A_rd_.resize(n_row_rd);
    B_rd_.resize(n_row_rd);
    C_rd_.resize(n_row_rd);
    D_rd_.resize(n_row_rd);

    A_rt_.resize(n_row_rt_);
    B_rt_.resize(n_row_rt_);
    C_rt_.resize(n_row_rt_);
    D_rt_.resize(n_row_rt_);
}

// ============================================================================
//  solve()
// ============================================================================

void PaScaLTDMASingle::solve(std::vector<double>& a, std::vector<double>& b,
                              std::vector<double>& c, std::vector<double>& d, int n_row) {
    if (nprocs_ == 1) {
        tdma_single(a, b, c, d, n_row);
        return;
    }

    // 1) Modified Thomas: eliminate lower diagonal
    a[0] = a[0] / b[0];
    d[0] = d[0] / b[0];
    c[0] = c[0] / b[0];

    a[1] = a[1] / b[1];
    d[1] = d[1] / b[1];
    c[1] = c[1] / b[1];

    for (int i = 2; i < n_row; ++i) {
        double r = 1.0 / (b[i] - a[i] * c[i - 1]);
        d[i] = r * (d[i] - a[i] * d[i - 1]);
        c[i] = r * c[i];
        a[i] = -r * a[i] * a[i - 1];
    }

    // 2) Modified Thomas: eliminate upper diagonal
    for (int i = n_row - 3; i >= 1; --i) {
        d[i] -= c[i] * d[i + 1];
        a[i] -= c[i] * a[i + 1];
        c[i] *= -c[i + 1];
    }

    double r = 1.0 / (1.0 - a[1] * c[0]);
    d[0] =  r * (d[0] - c[0] * d[1]);
    a[0] =  r * a[0];
    c[0] = -r * c[0] * c[1];

    // 3) Pack reduced system (boundary rows)
    A_rd_[0] = a[0];      A_rd_[1] = a[n_row - 1];
    B_rd_[0] = 1.0;       B_rd_[1] = 1.0;
    C_rd_[0] = c[0];      C_rd_[1] = c[n_row - 1];
    D_rd_[0] = d[0];      D_rd_[1] = d[n_row - 1];

    // 4) Gather all reduced rows to gather_rank
    {
        MPI_Request req[4];
        MPI_Igather(A_rd_.data(), 2, MPI_DOUBLE,
                    A_rt_.data(), 2, MPI_DOUBLE, gather_rank_, comm_, &req[0]);
        MPI_Igather(B_rd_.data(), 2, MPI_DOUBLE,
                    B_rt_.data(), 2, MPI_DOUBLE, gather_rank_, comm_, &req[1]);
        MPI_Igather(C_rd_.data(), 2, MPI_DOUBLE,
                    C_rt_.data(), 2, MPI_DOUBLE, gather_rank_, comm_, &req[2]);
        MPI_Igather(D_rd_.data(), 2, MPI_DOUBLE,
                    D_rt_.data(), 2, MPI_DOUBLE, gather_rank_, comm_, &req[3]);
        MPI_Waitall(4, req, MPI_STATUS_IGNORE);
    }

    // 5) Solve the gathered reduced system on gather_rank
    if (myrank_ == gather_rank_) {
        tdma_single(A_rt_, B_rt_, C_rt_, D_rt_, n_row_rt_);
    }

    // 6) Scatter solutions back
    {
        MPI_Request req[1];
        MPI_Iscatter(D_rt_.data(), 2, MPI_DOUBLE,
                     D_rd_.data(), 2, MPI_DOUBLE, gather_rank_, comm_, &req[0]);
        MPI_Wait(&req[0], MPI_STATUS_IGNORE);
    }

    // 7) Back-substitute local interior rows
    d[0]         = D_rd_[0];
    d[n_row - 1] = D_rd_[1];
    for (int i = 1; i < n_row - 1; ++i) {
        d[i] -= a[i] * d[0] + c[i] * d[n_row - 1];
    }
}

// ============================================================================
//  solve_cyclic()
// ============================================================================

void PaScaLTDMASingle::solve_cyclic(std::vector<double>& a, std::vector<double>& b,
                                     std::vector<double>& c, std::vector<double>& d, int n_row) {
    // 1) Modified Thomas: eliminate lower diagonal
    a[0] = a[0] / b[0];
    d[0] = d[0] / b[0];
    c[0] = c[0] / b[0];

    a[1] = a[1] / b[1];
    d[1] = d[1] / b[1];
    c[1] = c[1] / b[1];

    for (int i = 2; i < n_row; ++i) {
        double r = 1.0 / (b[i] - a[i] * c[i - 1]);
        d[i] = r * (d[i] - a[i] * d[i - 1]);
        c[i] = r * c[i];
        a[i] = -r * a[i] * a[i - 1];
    }

    // 2) Modified Thomas: eliminate upper diagonal
    for (int i = n_row - 3; i >= 1; --i) {
        d[i] -= c[i] * d[i + 1];
        a[i] -= c[i] * a[i + 1];
        c[i] *= -c[i + 1];
    }

    double r = 1.0 / (1.0 - a[1] * c[0]);
    d[0] =  r * (d[0] - c[0] * d[1]);
    a[0] =  r * a[0];
    c[0] = -r * c[0] * c[1];

    // 3) Pack reduced system (boundary rows)
    A_rd_[0] = a[0];      A_rd_[1] = a[n_row - 1];
    B_rd_[0] = 1.0;       B_rd_[1] = 1.0;
    C_rd_[0] = c[0];      C_rd_[1] = c[n_row - 1];
    D_rd_[0] = d[0];      D_rd_[1] = d[n_row - 1];

    // 4) Gather all reduced rows to gather_rank
    {
        MPI_Request req[4];
        MPI_Igather(A_rd_.data(), 2, MPI_DOUBLE,
                    A_rt_.data(), 2, MPI_DOUBLE, gather_rank_, comm_, &req[0]);
        MPI_Igather(B_rd_.data(), 2, MPI_DOUBLE,
                    B_rt_.data(), 2, MPI_DOUBLE, gather_rank_, comm_, &req[1]);
        MPI_Igather(C_rd_.data(), 2, MPI_DOUBLE,
                    C_rt_.data(), 2, MPI_DOUBLE, gather_rank_, comm_, &req[2]);
        MPI_Igather(D_rd_.data(), 2, MPI_DOUBLE,
                    D_rt_.data(), 2, MPI_DOUBLE, gather_rank_, comm_, &req[3]);
        MPI_Waitall(4, req, MPI_STATUS_IGNORE);
    }

    // 5) Solve the gathered reduced cyclic system on gather_rank
    if (myrank_ == gather_rank_) {
        tdma_cyclic_single(A_rt_, B_rt_, C_rt_, D_rt_, n_row_rt_);
    }

    // 6) Scatter solutions back
    {
        MPI_Request req[1];
        MPI_Iscatter(D_rt_.data(), 2, MPI_DOUBLE,
                     D_rd_.data(), 2, MPI_DOUBLE, gather_rank_, comm_, &req[0]);
        MPI_Wait(&req[0], MPI_STATUS_IGNORE);
    }

    // 7) Back-substitute local interior rows
    d[0]         = D_rd_[0];
    d[n_row - 1] = D_rd_[1];
    for (int i = 1; i < n_row - 1; ++i) {
        d[i] -= a[i] * d[0] + c[i] * d[n_row - 1];
    }
}
