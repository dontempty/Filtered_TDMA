#include "pascal_tdma_many_cuda.hpp"
#include "tdma_local_cuda.cuh"
#include "para_range.hpp"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ---------------------------------------------------------------------------
//  Small CUDA error helper
// ---------------------------------------------------------------------------
#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            std::fprintf(stderr,                                               \
                         "CUDA error %s at %s:%d: %s\n",                       \
                         cudaGetErrorName(_err), __FILE__, __LINE__,           \
                         cudaGetErrorString(_err));                            \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

namespace {

// Fill device buffer with a constant value (used to set B_rt to 1.0).
__global__ void fill_kernel(double* p, double v, std::size_t n) {
    std::size_t idx = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) p[idx] = v;
}

void cuda_fill(double* p, double v, std::size_t n) {
    if (n == 0) return;
    int block = 256;
    int grid  = (int)((n + block - 1) / block);
    fill_kernel<<<grid, block>>>(p, v, n);
}

} // namespace

// ===========================================================================
//  Constructor
// ===========================================================================
PaScaLTDMAManyCUDA::PaScaLTDMAManyCUDA(int n_sys, int myrank, int nprocs, MPI_Comm comm)
    : comm_(comm), nprocs_(nprocs), n_sys_(n_sys)
{
    const int n_row_rd = 2;

    int ista, iend;
    para_range(1, n_sys, nprocs, myrank, ista, iend);
    n_sys_rt_ = iend - ista + 1;

    std::vector<int> ns_rt_array(nprocs);
    MPI_Allgather(&n_sys_rt_, 1, MPI_INT,
                  ns_rt_array.data(), 1, MPI_INT,
                  comm);

    n_row_rt_ = n_row_rd * nprocs;

    // Reduced-system device buffers [2 × n_sys]
    CUDA_CHECK(cudaMalloc(&d_A_rd_, sizeof(double) * (std::size_t)n_row_rd * n_sys));
    CUDA_CHECK(cudaMalloc(&d_B_rd_, sizeof(double) * (std::size_t)n_row_rd * n_sys));
    CUDA_CHECK(cudaMalloc(&d_C_rd_, sizeof(double) * (std::size_t)n_row_rd * n_sys));
    CUDA_CHECK(cudaMalloc(&d_D_rd_, sizeof(double) * (std::size_t)n_row_rd * n_sys));

    // Transposed device buffers [n_row_rt × n_sys_rt]
    std::size_t rt_sz = (std::size_t)n_row_rt_ * (std::size_t)n_sys_rt_;
    CUDA_CHECK(cudaMalloc(&d_A_rt_, sizeof(double) * rt_sz));
    CUDA_CHECK(cudaMalloc(&d_B_rt_, sizeof(double) * rt_sz));
    CUDA_CHECK(cudaMalloc(&d_C_rt_, sizeof(double) * rt_sz));
    CUDA_CHECK(cudaMalloc(&d_D_rt_, sizeof(double) * rt_sz));
    CUDA_CHECK(cudaMalloc(&d_E_rt_, sizeof(double) * rt_sz));

    // B_rt is identity diagonal (always 1.0) — set once.
    cuda_fill(d_B_rt_, 1.0, rt_sz);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Build derived datatypes (identical to CPU version)
    ddtype_Fs_.resize(nprocs);
    ddtype_Bs_.resize(nprocs);
    int bigsize[2], subsize[2], start[2];
    int col_offset = 0;
    for (int p = 0; p < nprocs; ++p) {
        bigsize[0] = n_row_rd; bigsize[1] = n_sys;
        subsize[0] = n_row_rd; subsize[1] = ns_rt_array[p];
        start[0]   = 0;        start[1]   = col_offset;
        col_offset += ns_rt_array[p];
        MPI_Type_create_subarray(2, bigsize, subsize, start,
                                 MPI_ORDER_C, MPI_DOUBLE, &ddtype_Fs_[p]);
        MPI_Type_commit(&ddtype_Fs_[p]);

        bigsize[0] = n_row_rt_; bigsize[1] = n_sys_rt_;
        subsize[0] = n_row_rd;  subsize[1] = n_sys_rt_;
        start[0]   = n_row_rd * p; start[1] = 0;
        MPI_Type_create_subarray(2, bigsize, subsize, start,
                                 MPI_ORDER_C, MPI_DOUBLE, &ddtype_Bs_[p]);
        MPI_Type_commit(&ddtype_Bs_[p]);
    }

    count_send_.assign(nprocs, 1);
    displ_send_.assign(nprocs, 0);
    count_recv_.assign(nprocs, 1);
    displ_recv_.assign(nprocs, 0);
}

// ===========================================================================
//  Destructor
// ===========================================================================
PaScaLTDMAManyCUDA::~PaScaLTDMAManyCUDA() {
    for (int p = 0; p < nprocs_; ++p) {
        if (ddtype_Fs_[p] != MPI_DATATYPE_NULL) MPI_Type_free(&ddtype_Fs_[p]);
        if (ddtype_Bs_[p] != MPI_DATATYPE_NULL) MPI_Type_free(&ddtype_Bs_[p]);
    }

    auto safe_free = [](double*& p) {
        if (p) { cudaFree(p); p = nullptr; }
    };
    safe_free(d_A_rd_); safe_free(d_B_rd_); safe_free(d_C_rd_); safe_free(d_D_rd_);
    safe_free(d_A_rt_); safe_free(d_B_rt_); safe_free(d_C_rt_); safe_free(d_D_rt_);
    safe_free(d_E_rt_); safe_free(d_E_loc_);
}

// ===========================================================================
//  Lazy allocation of cyclic local workspace
// ===========================================================================
void PaScaLTDMAManyCUDA::ensure_E_loc(int n_sys, int n_row) {
    std::size_t need = (std::size_t)n_row * (std::size_t)n_sys;
    if (need <= e_loc_capacity_) return;
    if (d_E_loc_) cudaFree(d_E_loc_);
    CUDA_CHECK(cudaMalloc(&d_E_loc_, sizeof(double) * need));
    e_loc_capacity_ = need;
}

// ===========================================================================
//  solve()
// ===========================================================================
void PaScaLTDMAManyCUDA::solve(double* d_A, double* d_B, double* d_C, double* d_D,
                               int n_sys, int n_row) {
    if (nprocs_ == 1) {
        tdma_many_cuda(d_A, d_B, d_C, d_D, n_sys, n_row);
        return;
    }

    // 1+2+3) Modified Thomas — produces reduced [2 × n_sys] system in *_rd
    pascal_tdma_modified_thomas_cuda(d_A, d_B, d_C, d_D,
                                     d_A_rd_, d_B_rd_, d_C_rd_, d_D_rd_,
                                     n_sys, n_row);

    // Make the kernel results visible to MPI before alltoall
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4) Alltoall: scatter reduced rows so each rank holds the global reduced
    //    system for its column subset. Operates on device pointers (CUDA-aware MPI).
    {
        MPI_Request req[3];
        MPI_Ialltoallw(d_A_rd_, count_send_.data(), displ_send_.data(), ddtype_Fs_.data(),
                       d_A_rt_, count_recv_.data(), displ_recv_.data(), ddtype_Bs_.data(),
                       comm_, &req[0]);
        MPI_Ialltoallw(d_C_rd_, count_send_.data(), displ_send_.data(), ddtype_Fs_.data(),
                       d_C_rt_, count_recv_.data(), displ_recv_.data(), ddtype_Bs_.data(),
                       comm_, &req[1]);
        MPI_Ialltoallw(d_D_rd_, count_send_.data(), displ_send_.data(), ddtype_Fs_.data(),
                       d_D_rt_, count_recv_.data(), displ_recv_.data(), ddtype_Bs_.data(),
                       comm_, &req[2]);
        MPI_Waitall(3, req, MPI_STATUSES_IGNORE);
    }

    // 5) Local solve on the reduced system
    tdma_many_cuda(d_A_rt_, d_B_rt_, d_C_rt_, d_D_rt_, n_sys_rt_, n_row_rt_);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 6) Alltoall back: gather solutions into D_rd
    {
        MPI_Request req[1];
        MPI_Ialltoallw(d_D_rt_, count_recv_.data(), displ_recv_.data(), ddtype_Bs_.data(),
                       d_D_rd_, count_send_.data(), displ_send_.data(), ddtype_Fs_.data(),
                       comm_, &req[0]);
        MPI_Wait(&req[0], MPI_STATUS_IGNORE);
    }

    // 7) Final update — fill boundaries and propagate to interior rows
    pascal_tdma_update_solution_cuda(d_A, d_C, d_D, d_D_rd_, n_sys, n_row);
}

// ===========================================================================
//  solve_cyclic()
// ===========================================================================
void PaScaLTDMAManyCUDA::solve_cyclic(double* d_A, double* d_B, double* d_C, double* d_D,
                                      int n_sys, int n_row) {
    if (nprocs_ == 1) {
        ensure_E_loc(n_sys, n_row);
        tdma_cyclic_many_cuda(d_A, d_B, d_C, d_D, d_E_loc_, n_sys, n_row);
        return;
    }

    pascal_tdma_modified_thomas_cuda(d_A, d_B, d_C, d_D,
                                     d_A_rd_, d_B_rd_, d_C_rd_, d_D_rd_,
                                     n_sys, n_row);
    CUDA_CHECK(cudaDeviceSynchronize());

    {
        MPI_Request req[3];
        MPI_Ialltoallw(d_A_rd_, count_send_.data(), displ_send_.data(), ddtype_Fs_.data(),
                       d_A_rt_, count_recv_.data(), displ_recv_.data(), ddtype_Bs_.data(),
                       comm_, &req[0]);
        MPI_Ialltoallw(d_C_rd_, count_send_.data(), displ_send_.data(), ddtype_Fs_.data(),
                       d_C_rt_, count_recv_.data(), displ_recv_.data(), ddtype_Bs_.data(),
                       comm_, &req[1]);
        MPI_Ialltoallw(d_D_rd_, count_send_.data(), displ_send_.data(), ddtype_Fs_.data(),
                       d_D_rt_, count_recv_.data(), displ_recv_.data(), ddtype_Bs_.data(),
                       comm_, &req[2]);
        MPI_Waitall(3, req, MPI_STATUSES_IGNORE);
    }

    // Cyclic solve on the reduced system; d_E_rt_ is the Sherman-Morrison workspace
    tdma_cyclic_many_cuda(d_A_rt_, d_B_rt_, d_C_rt_, d_D_rt_, d_E_rt_,
                          n_sys_rt_, n_row_rt_);
    CUDA_CHECK(cudaDeviceSynchronize());

    {
        MPI_Request req[1];
        MPI_Ialltoallw(d_D_rt_, count_recv_.data(), displ_recv_.data(), ddtype_Bs_.data(),
                       d_D_rd_, count_send_.data(), displ_send_.data(), ddtype_Fs_.data(),
                       comm_, &req[0]);
        MPI_Wait(&req[0], MPI_STATUS_IGNORE);
    }

    pascal_tdma_update_solution_cuda(d_A, d_C, d_D, d_D_rd_, n_sys, n_row);
}
