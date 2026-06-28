#include "../include/pascal_tdma_many.hpp"
#include "tdma_local.hpp"
#include "para_range.hpp"

#include <cstddef>
#include <vector>

// ============================================================================
//  Constructor / Destructor
// ============================================================================

PaScaLTDMAMany::PaScaLTDMAMany(int n_sys, int myrank, int nprocs, MPI_Comm comm)
    : comm_(comm), nprocs_(nprocs)
{
    // Reduced system dimensions: each rank contributes 2 boundary rows
    const int n_row_rd = 2;

    // Determine how many systems this rank owns after the alltoall transpose
    int ista, iend;
    para_range(1, n_sys, nprocs, myrank, ista, iend);
    n_sys_rt_ = iend - ista + 1;

    // Gather the per-rank system counts across all ranks
    std::vector<int> ns_rt_array(nprocs);
    MPI_Allgather(&n_sys_rt_, 1, MPI_INT,
                  ns_rt_array.data(), 1, MPI_INT,
                  comm);

    n_row_rt_ = n_row_rd * nprocs;   // reduced rows after transpose

    // Allocate reduced system buffers
    A_rd_.resize(n_row_rd * n_sys);
    B_rd_.resize(n_row_rd * n_sys);
    C_rd_.resize(n_row_rd * n_sys);
    D_rd_.resize(n_row_rd * n_sys);

    // Allocate transposed system buffers (B is always 1)
    A_rt_.resize(n_row_rt_ * n_sys_rt_);
    B_rt_.resize(n_row_rt_ * n_sys_rt_, 1.0);
    C_rt_.resize(n_row_rt_ * n_sys_rt_);
    D_rt_.resize(n_row_rt_ * n_sys_rt_);

    // Build MPI derived datatypes for the alltoall pattern
    ddtype_Fs_.resize(nprocs);
    ddtype_Bs_.resize(nprocs);
    std::vector<int> bigsize(2), subsize(2), start(2);
    int col_offset = 0;
    for (int p = 0; p < nprocs; ++p) {
        // Send: subarray of [n_row_rd × n_sys] — slice ns_rt_array[p] columns
        bigsize[0] = n_row_rd; bigsize[1] = n_sys;
        subsize[0] = n_row_rd; subsize[1] = ns_rt_array[p];
        start[0]   = 0;        start[1]   = col_offset;
        col_offset += ns_rt_array[p];
        MPI_Type_create_subarray(2, bigsize.data(), subsize.data(), start.data(),
                                 MPI_ORDER_C, MPI_DOUBLE, &ddtype_Fs_[p]);
        MPI_Type_commit(&ddtype_Fs_[p]);

        // Recv: subarray of [n_row_rt × n_sys_rt] — slice n_row_rd rows starting at p*n_row_rd
        bigsize[0] = n_row_rt_; bigsize[1] = n_sys_rt_;
        subsize[0] = n_row_rd;  subsize[1] = n_sys_rt_;
        start[0]   = n_row_rd * p; start[1] = 0;
        MPI_Type_create_subarray(2, bigsize.data(), subsize.data(), start.data(),
                                 MPI_ORDER_C, MPI_DOUBLE, &ddtype_Bs_[p]);
        MPI_Type_commit(&ddtype_Bs_[p]);
    }

    // All counts are 1 and displacements are 0 because the DDTs encode the layout
    count_send_.assign(nprocs, 1);
    displ_send_.assign(nprocs, 0);
    count_recv_.assign(nprocs, 1);
    displ_recv_.assign(nprocs, 0);
}

PaScaLTDMAMany::~PaScaLTDMAMany() {
    for (int p = 0; p < nprocs_; ++p) {
        if (ddtype_Fs_[p] != MPI_DATATYPE_NULL) MPI_Type_free(&ddtype_Fs_[p]);
        if (ddtype_Bs_[p] != MPI_DATATYPE_NULL) MPI_Type_free(&ddtype_Bs_[p]);
    }
}

// ============================================================================
//  Helper: boundary-row pointer setup
// ============================================================================

struct BoundaryPtrs {
    double *A0, *A1, *AN;
    double *B0, *B1;
    double *C0, *C1, *CN;
    double *D0, *D1, *DN;
};

static inline BoundaryPtrs setup_ptrs(double* A, double* B, double* C, double* D,
                                       int n_sys, int n_row) {
    BoundaryPtrs p;
    p.A0 = A;
    p.A1 = A + (std::size_t)1 * n_sys;
    p.AN = A + (std::size_t)(n_row - 1) * n_sys;
    p.B0 = B;
    p.B1 = B + (std::size_t)1 * n_sys;
    p.C0 = C;
    p.C1 = C + (std::size_t)1 * n_sys;
    p.CN = C + (std::size_t)(n_row - 1) * n_sys;
    p.D0 = D;
    p.D1 = D + (std::size_t)1 * n_sys;
    p.DN = D + (std::size_t)(n_row - 1) * n_sys;
    return p;
}

// ============================================================================
//  solve() — no timing
// ============================================================================

void PaScaLTDMAMany::solve(double* __restrict A, double* __restrict B,
                           double* __restrict C, double* __restrict D,
                           int n_sys, int n_row) {
    if (nprocs_ == 1) {
        tdma_many(A, B, C, D, n_sys, n_row);
        return;
    }

    int i, j;
    auto p = setup_ptrs(A, B, C, D, n_sys, n_row);

    double* A_rd0 = A_rd_.data();
    double* A_rd1 = A_rd_.data() + n_sys;
    double* C_rd0 = C_rd_.data();
    double* C_rd1 = C_rd_.data() + n_sys;
    double* D_rd0 = D_rd_.data();
    double* D_rd1 = D_rd_.data() + n_sys;

    // 1) Forward Elimination
    #pragma omp simd
    for (i = 0; i < n_sys; ++i) {
        double r0 = 1.0 / p.B0[i];
        p.A0[i] *= r0; p.C0[i] *= r0; p.D0[i] *= r0;
        double r1 = 1.0 / p.B1[i];
        p.A1[i] *= r1; p.C1[i] *= r1; p.D1[i] *= r1;
    }
    for (j = 2; j < n_row; ++j) {
        double*       Aj  = A + (std::size_t)j * n_sys;
        double*       Bj  = B + (std::size_t)j * n_sys;
        double*       Cj  = C + (std::size_t)j * n_sys;
        double*       Dj  = D + (std::size_t)j * n_sys;
        const double* Ajm = A + (std::size_t)(j - 1) * n_sys;
        const double* Cjm = C + (std::size_t)(j - 1) * n_sys;
        const double* Djm = D + (std::size_t)(j - 1) * n_sys;
        #pragma omp simd
        for (i = 0; i < n_sys; ++i) {
            double inv = 1.0 / (Bj[i] - Aj[i] * Cjm[i]);
            Dj[i] =  inv * (Dj[i] - Aj[i] * Djm[i]);
            Cj[i] =  inv * Cj[i];
            Aj[i] = -inv * Aj[i] * Ajm[i];
        }
    }

    // 2) Backward Substitution
    for (j = n_row - 3; j >= 1; --j) {
        double*       Aj  = A + (std::size_t)j * n_sys;
        double*       Cj  = C + (std::size_t)j * n_sys;
        double*       Dj  = D + (std::size_t)j * n_sys;
        const double* Ajp = A + (std::size_t)(j + 1) * n_sys;
        const double* Cjp = C + (std::size_t)(j + 1) * n_sys;
        const double* Djp = D + (std::size_t)(j + 1) * n_sys;
        #pragma omp simd
        for (i = 0; i < n_sys; ++i) {
            Dj[i] -= Cj[i] * Djp[i];
            Aj[i] -= Cj[i] * Ajp[i];
            Cj[i] *= -Cjp[i];
        }
    }

    // 3) Pack reduced system (boundary rows 0 and n_row-1)
    #pragma omp simd
    for (i = 0; i < n_sys; ++i) {
        double r = 1.0 / (1.0 - p.A1[i] * p.C0[i]);
        p.D0[i] =  r * (p.D0[i] - p.C0[i] * p.D1[i]);
        p.A0[i] =  r * p.A0[i];
        p.C0[i] = -r * p.C0[i] * p.C1[i];

        A_rd0[i] = p.A0[i];  A_rd1[i] = p.AN[i];
        C_rd0[i] = p.C0[i];  C_rd1[i] = p.CN[i];
        D_rd0[i] = p.D0[i];  D_rd1[i] = p.DN[i];
    }

    // 4) MPI alltoall: transpose the reduced system
    {
        MPI_Request req[3];
        MPI_Ialltoallw(A_rd_.data(), count_send_.data(), displ_send_.data(), ddtype_Fs_.data(),
                       A_rt_.data(), count_recv_.data(), displ_recv_.data(), ddtype_Bs_.data(),
                       comm_, &req[0]);
        MPI_Ialltoallw(C_rd_.data(), count_send_.data(), displ_send_.data(), ddtype_Fs_.data(),
                       C_rt_.data(), count_recv_.data(), displ_recv_.data(), ddtype_Bs_.data(),
                       comm_, &req[1]);
        MPI_Ialltoallw(D_rd_.data(), count_send_.data(), displ_send_.data(), ddtype_Fs_.data(),
                       D_rt_.data(), count_recv_.data(), displ_recv_.data(), ddtype_Bs_.data(),
                       comm_, &req[2]);
        MPI_Waitall(3, req, MPI_STATUSES_IGNORE);
    }

    // 5) Solve the local reduced system
    tdma_many(A_rt_.data(), B_rt_.data(), C_rt_.data(), D_rt_.data(), n_sys_rt_, n_row_rt_);

    // 6) Alltoall back: scatter solutions to owning ranks
    {
        MPI_Request req[1];
        MPI_Ialltoallw(D_rt_.data(), count_recv_.data(), displ_recv_.data(), ddtype_Bs_.data(),
                       D_rd_.data(), count_send_.data(), displ_send_.data(), ddtype_Fs_.data(),
                       comm_, &req[0]);
        MPI_Wait(&req[0], MPI_STATUS_IGNORE);
    }

    // 7) Final local solve
    #pragma omp simd
    for (i = 0; i < n_sys; ++i) {
        p.D0[i] = D_rd0[i];
        p.DN[i] = D_rd1[i];
    }
    for (j = 1; j < n_row - 1; ++j) {
        double*       Dj = D + (std::size_t)j * n_sys;
        const double* Aj = A + (std::size_t)j * n_sys;
        const double* Cj = C + (std::size_t)j * n_sys;
        #pragma omp simd
        for (i = 0; i < n_sys; ++i) {
            Dj[i] -= Aj[i] * p.D0[i] + Cj[i] * p.DN[i];
        }
    }
}

// ============================================================================
//  solve_cyclic() — like solve() but step 5 uses tdma_cyclic_many
//  so the global reduced system is solved as a cyclic tridiagonal.
// ============================================================================

void PaScaLTDMAMany::solve_cyclic(double* __restrict A, double* __restrict B,
                                  double* __restrict C, double* __restrict D,
                                  int n_sys, int n_row) {
    if (nprocs_ == 1) {
        tdma_cyclic_many(A, B, C, D, n_sys, n_row);
        return;
    }

    int i, j;
    auto p = setup_ptrs(A, B, C, D, n_sys, n_row);

    double* A_rd0 = A_rd_.data();
    double* A_rd1 = A_rd_.data() + n_sys;
    double* C_rd0 = C_rd_.data();
    double* C_rd1 = C_rd_.data() + n_sys;
    double* D_rd0 = D_rd_.data();
    double* D_rd1 = D_rd_.data() + n_sys;

    // 1) Forward Elimination
    #pragma omp simd
    for (i = 0; i < n_sys; ++i) {
        double r0 = 1.0 / p.B0[i];
        p.A0[i] *= r0; p.C0[i] *= r0; p.D0[i] *= r0;
        double r1 = 1.0 / p.B1[i];
        p.A1[i] *= r1; p.C1[i] *= r1; p.D1[i] *= r1;
    }
    for (j = 2; j < n_row; ++j) {
        double*       Aj  = A + (std::size_t)j * n_sys;
        double*       Bj  = B + (std::size_t)j * n_sys;
        double*       Cj  = C + (std::size_t)j * n_sys;
        double*       Dj  = D + (std::size_t)j * n_sys;
        const double* Ajm = A + (std::size_t)(j - 1) * n_sys;
        const double* Cjm = C + (std::size_t)(j - 1) * n_sys;
        const double* Djm = D + (std::size_t)(j - 1) * n_sys;
        #pragma omp simd
        for (i = 0; i < n_sys; ++i) {
            double inv = 1.0 / (Bj[i] - Aj[i] * Cjm[i]);
            Dj[i] =  inv * (Dj[i] - Aj[i] * Djm[i]);
            Cj[i] =  inv * Cj[i];
            Aj[i] = -inv * Aj[i] * Ajm[i];
        }
    }

    // 2) Backward Substitution
    for (j = n_row - 3; j >= 1; --j) {
        double*       Aj  = A + (std::size_t)j * n_sys;
        double*       Cj  = C + (std::size_t)j * n_sys;
        double*       Dj  = D + (std::size_t)j * n_sys;
        const double* Ajp = A + (std::size_t)(j + 1) * n_sys;
        const double* Cjp = C + (std::size_t)(j + 1) * n_sys;
        const double* Djp = D + (std::size_t)(j + 1) * n_sys;
        #pragma omp simd
        for (i = 0; i < n_sys; ++i) {
            Dj[i] -= Cj[i] * Djp[i];
            Aj[i] -= Cj[i] * Ajp[i];
            Cj[i] *= -Cjp[i];
        }
    }

    // 3) Pack reduced system (boundary rows 0 and n_row-1)
    #pragma omp simd
    for (i = 0; i < n_sys; ++i) {
        double r = 1.0 / (1.0 - p.A1[i] * p.C0[i]);
        p.D0[i] =  r * (p.D0[i] - p.C0[i] * p.D1[i]);
        p.A0[i] =  r * p.A0[i];
        p.C0[i] = -r * p.C0[i] * p.C1[i];

        A_rd0[i] = p.A0[i];  A_rd1[i] = p.AN[i];
        C_rd0[i] = p.C0[i];  C_rd1[i] = p.CN[i];
        D_rd0[i] = p.D0[i];  D_rd1[i] = p.DN[i];
    }

    // 4) MPI alltoall: transpose the reduced system
    {
        MPI_Request req[3];
        MPI_Ialltoallw(A_rd_.data(), count_send_.data(), displ_send_.data(), ddtype_Fs_.data(),
                       A_rt_.data(), count_recv_.data(), displ_recv_.data(), ddtype_Bs_.data(),
                       comm_, &req[0]);
        MPI_Ialltoallw(C_rd_.data(), count_send_.data(), displ_send_.data(), ddtype_Fs_.data(),
                       C_rt_.data(), count_recv_.data(), displ_recv_.data(), ddtype_Bs_.data(),
                       comm_, &req[1]);
        MPI_Ialltoallw(D_rd_.data(), count_send_.data(), displ_send_.data(), ddtype_Fs_.data(),
                       D_rt_.data(), count_recv_.data(), displ_recv_.data(), ddtype_Bs_.data(),
                       comm_, &req[2]);
        MPI_Waitall(3, req, MPI_STATUSES_IGNORE);
    }

    // 5) Solve the local reduced system — cyclic version
    tdma_cyclic_many(A_rt_.data(), B_rt_.data(), C_rt_.data(), D_rt_.data(), n_sys_rt_, n_row_rt_);

    // 6) Alltoall back: scatter solutions to owning ranks
    {
        MPI_Request req[1];
        MPI_Ialltoallw(D_rt_.data(), count_recv_.data(), displ_recv_.data(), ddtype_Bs_.data(),
                       D_rd_.data(), count_send_.data(), displ_send_.data(), ddtype_Fs_.data(),
                       comm_, &req[0]);
        MPI_Wait(&req[0], MPI_STATUS_IGNORE);
    }

    // 7) Final local solve
    #pragma omp simd
    for (i = 0; i < n_sys; ++i) {
        p.D0[i] = D_rd0[i];
        p.DN[i] = D_rd1[i];
    }
    for (j = 1; j < n_row - 1; ++j) {
        double*       Dj = D + (std::size_t)j * n_sys;
        const double* Aj = A + (std::size_t)j * n_sys;
        const double* Cj = C + (std::size_t)j * n_sys;
        #pragma omp simd
        for (i = 0; i < n_sys; ++i) {
            Dj[i] -= Aj[i] * p.D0[i] + Cj[i] * p.DN[i];
        }
    }
}

// ============================================================================
//  solve_profile() — per-phase timing with MPI_Barrier (7 entries)
// ============================================================================

void PaScaLTDMAMany::solve_profile(double* __restrict A, double* __restrict B,
                                   double* __restrict C, double* __restrict D,
                                   int n_sys, int n_row,
                                   std::vector<double>& time_list) {
    time_list.resize(7);
    if (nprocs_ == 1) {
        double t0 = MPI_Wtime();
        tdma_many(A, B, C, D, n_sys, n_row);
        time_list.assign(7, 0.0);
        time_list[0] = MPI_Wtime() - t0;
        return;
    }

    int i, j;
    double t0, t1;
    auto p = setup_ptrs(A, B, C, D, n_sys, n_row);

    double* A_rd0 = A_rd_.data();
    double* A_rd1 = A_rd_.data() + n_sys;
    double* C_rd0 = C_rd_.data();
    double* C_rd1 = C_rd_.data() + n_sys;
    double* D_rd0 = D_rd_.data();
    double* D_rd1 = D_rd_.data() + n_sys;

    // 1) Forward Elimination
    MPI_Barrier(comm_); t0 = MPI_Wtime();
    #pragma omp simd
    for (i = 0; i < n_sys; ++i) {
        double r0 = 1.0 / p.B0[i];
        p.A0[i] *= r0; p.C0[i] *= r0; p.D0[i] *= r0;
        double r1 = 1.0 / p.B1[i];
        p.A1[i] *= r1; p.C1[i] *= r1; p.D1[i] *= r1;
    }
    for (j = 2; j < n_row; ++j) {
        double*       Aj  = A + (std::size_t)j * n_sys;
        double*       Bj  = B + (std::size_t)j * n_sys;
        double*       Cj  = C + (std::size_t)j * n_sys;
        double*       Dj  = D + (std::size_t)j * n_sys;
        const double* Ajm = A + (std::size_t)(j - 1) * n_sys;
        const double* Cjm = C + (std::size_t)(j - 1) * n_sys;
        const double* Djm = D + (std::size_t)(j - 1) * n_sys;
        #pragma omp simd
        for (i = 0; i < n_sys; ++i) {
            double inv = 1.0 / (Bj[i] - Aj[i] * Cjm[i]);
            Dj[i] =  inv * (Dj[i] - Aj[i] * Djm[i]);
            Cj[i] =  inv * Cj[i];
            Aj[i] = -inv * Aj[i] * Ajm[i];
        }
    }
    t1 = MPI_Wtime(); time_list[0] = t1 - t0;

    // 2) Backward Substitution
    MPI_Barrier(comm_); t0 = MPI_Wtime();
    for (j = n_row - 3; j >= 1; --j) {
        double*       Aj  = A + (std::size_t)j * n_sys;
        double*       Cj  = C + (std::size_t)j * n_sys;
        double*       Dj  = D + (std::size_t)j * n_sys;
        const double* Ajp = A + (std::size_t)(j + 1) * n_sys;
        const double* Cjp = C + (std::size_t)(j + 1) * n_sys;
        const double* Djp = D + (std::size_t)(j + 1) * n_sys;
        #pragma omp simd
        for (i = 0; i < n_sys; ++i) {
            Dj[i] -= Cj[i] * Djp[i];
            Aj[i] -= Cj[i] * Ajp[i];
            Cj[i] *= -Cjp[i];
        }
    }
    t1 = MPI_Wtime(); time_list[1] = t1 - t0;

    // 3) Pack
    MPI_Barrier(comm_); t0 = MPI_Wtime();
    #pragma omp simd
    for (i = 0; i < n_sys; ++i) {
        double r = 1.0 / (1.0 - p.A1[i] * p.C0[i]);
        p.D0[i] =  r * (p.D0[i] - p.C0[i] * p.D1[i]);
        p.A0[i] =  r * p.A0[i];
        p.C0[i] = -r * p.C0[i] * p.C1[i];

        A_rd0[i] = p.A0[i];  A_rd1[i] = p.AN[i];
        C_rd0[i] = p.C0[i];  C_rd1[i] = p.CN[i];
        D_rd0[i] = p.D0[i];  D_rd1[i] = p.DN[i];
    }
    t1 = MPI_Wtime(); time_list[2] = t1 - t0;

    // 4) MPI alltoall (scatter)
    MPI_Barrier(comm_); t0 = MPI_Wtime();
    {
        MPI_Request req[3];
        MPI_Ialltoallw(A_rd_.data(), count_send_.data(), displ_send_.data(), ddtype_Fs_.data(),
                       A_rt_.data(), count_recv_.data(), displ_recv_.data(), ddtype_Bs_.data(),
                       comm_, &req[0]);
        MPI_Ialltoallw(C_rd_.data(), count_send_.data(), displ_send_.data(), ddtype_Fs_.data(),
                       C_rt_.data(), count_recv_.data(), displ_recv_.data(), ddtype_Bs_.data(),
                       comm_, &req[1]);
        MPI_Ialltoallw(D_rd_.data(), count_send_.data(), displ_send_.data(), ddtype_Fs_.data(),
                       D_rt_.data(), count_recv_.data(), displ_recv_.data(), ddtype_Bs_.data(),
                       comm_, &req[2]);
        MPI_Waitall(3, req, MPI_STATUSES_IGNORE);
    }
    t1 = MPI_Wtime(); time_list[3] = t1 - t0;

    // 5) Solve local reduced system
    MPI_Barrier(comm_); t0 = MPI_Wtime();
    tdma_many(A_rt_.data(), B_rt_.data(), C_rt_.data(), D_rt_.data(), n_sys_rt_, n_row_rt_);
    t1 = MPI_Wtime(); time_list[4] = t1 - t0;

    // 6) MPI alltoall (gather)
    MPI_Barrier(comm_); t0 = MPI_Wtime();
    {
        MPI_Request req[1];
        MPI_Ialltoallw(D_rt_.data(), count_recv_.data(), displ_recv_.data(), ddtype_Bs_.data(),
                       D_rd_.data(), count_send_.data(), displ_send_.data(), ddtype_Fs_.data(),
                       comm_, &req[0]);
        MPI_Wait(&req[0], MPI_STATUS_IGNORE);
    }
    t1 = MPI_Wtime(); time_list[5] = t1 - t0;

    // 7) Final local solve
    MPI_Barrier(comm_); t0 = MPI_Wtime();
    #pragma omp simd
    for (i = 0; i < n_sys; ++i) {
        p.D0[i] = D_rd0[i];
        p.DN[i] = D_rd1[i];
    }
    for (j = 1; j < n_row - 1; ++j) {
        double*       Dj = D + (std::size_t)j * n_sys;
        const double* Aj = A + (std::size_t)j * n_sys;
        const double* Cj = C + (std::size_t)j * n_sys;
        #pragma omp simd
        for (i = 0; i < n_sys; ++i) {
            Dj[i] -= Aj[i] * p.D0[i] + Cj[i] * p.DN[i];
        }
    }
    t1 = MPI_Wtime(); time_list[6] = t1 - t0;
}
