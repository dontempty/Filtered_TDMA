#include "../include/pascal_tdma_cuda.hpp"
#include "pascal_tdma_kernels.cuh"
#include "nvtx_util.hpp"
#include "tdma_local_cuda.cuh"
#include "../cpu/para_range.hpp"

#include <cuda_runtime.h>
#include <algorithm>
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

// ===========================================================================
//  Constructor
// ===========================================================================
PaScaLTDMACUDA::PaScaLTDMACUDA(int n_sys, int myrank, int nprocs, MPI_Comm comm,
                                       int block_x, int block_y)
    : comm_(comm), myrank_(myrank), nprocs_(nprocs), n_sys_(n_sys),
      block_x_(block_x), block_y_(block_y)
{
    int ista, iend;
    para_range(1, n_sys, nprocs, myrank, ista, iend);
    n_sys_rt_ = iend - ista + 1;

    h_ns_rt_.assign(nprocs, 0);
    MPI_Allgather(&n_sys_rt_, 1, MPI_INT,
                  h_ns_rt_.data(), 1, MPI_INT,
                  comm);

    h_col_offset_.assign(nprocs, 0);
    for (int p = 1; p < nprocs; ++p)
        h_col_offset_[p] = h_col_offset_[p - 1] + h_ns_rt_[p - 1];

    n_row_rt_ = N_ROW_RD * nprocs;

    // Pre-compute Alltoallv counts/displs in MPI_DOUBLE units.
    send_counts_fwd_.assign(nprocs, 0);
    send_displs_fwd_.assign(nprocs, 0);
    recv_counts_fwd_.assign(nprocs, 0);
    recv_displs_fwd_.assign(nprocs, 0);
    send_counts_bwd_.assign(nprocs, 0);
    send_displs_bwd_.assign(nprocs, 0);
    recv_counts_bwd_.assign(nprocs, 0);
    recv_displs_bwd_.assign(nprocs, 0);

    int fwd_send_off = 0, fwd_recv_off = 0;
    int bwd_send_off = 0, bwd_recv_off = 0;
    for (int p = 0; p < nprocs; ++p) {
        // Forward: send to rank p = 2 × ns_rt[p];  recv from rank p = 2 × n_sys_rt_ (mine)
        send_counts_fwd_[p] = N_ROW_RD * h_ns_rt_[p];
        send_displs_fwd_[p] = fwd_send_off;
        fwd_send_off       += send_counts_fwd_[p];

        recv_counts_fwd_[p] = N_ROW_RD * n_sys_rt_;
        recv_displs_fwd_[p] = fwd_recv_off;
        fwd_recv_off       += recv_counts_fwd_[p];

        // Backward is the reverse — counts/displs swap roles.
        send_counts_bwd_[p] = N_ROW_RD * n_sys_rt_;
        send_displs_bwd_[p] = bwd_send_off;
        bwd_send_off       += send_counts_bwd_[p];

        recv_counts_bwd_[p] = N_ROW_RD * h_ns_rt_[p];
        recv_displs_bwd_[p] = bwd_recv_off;
        bwd_recv_off       += recv_counts_bwd_[p];
    }

    // Reduced-system device buffers [2 × n_sys]
    CUDA_CHECK(cudaMalloc(&d_A_rd_, sizeof(double) * (std::size_t)N_ROW_RD * n_sys));
    CUDA_CHECK(cudaMalloc(&d_B_rd_, sizeof(double) * (std::size_t)N_ROW_RD * n_sys));
    CUDA_CHECK(cudaMalloc(&d_C_rd_, sizeof(double) * (std::size_t)N_ROW_RD * n_sys));
    CUDA_CHECK(cudaMalloc(&d_D_rd_, sizeof(double) * (std::size_t)N_ROW_RD * n_sys));

    // Transposed device buffers [n_row_rt × n_sys_rt]
    std::size_t rt_sz = (std::size_t)n_row_rt_ * (std::size_t)n_sys_rt_;
    CUDA_CHECK(cudaMalloc(&d_A_rt_, sizeof(double) * rt_sz));
    CUDA_CHECK(cudaMalloc(&d_B_rt_, sizeof(double) * rt_sz));
    CUDA_CHECK(cudaMalloc(&d_C_rt_, sizeof(double) * rt_sz));
    CUDA_CHECK(cudaMalloc(&d_D_rt_, sizeof(double) * rt_sz));
    CUDA_CHECK(cudaMalloc(&d_E_rt_, sizeof(double) * rt_sz));

    // B_rt is identity diagonal (always 1.0) — set once. b_rd never needs transport.
    pascal_cuda_fill(d_B_rt_, 1.0, rt_sz);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Pack/unpack scratch buffers for backward alltoall (D only).
    std::size_t fwd_send = (std::size_t)N_ROW_RD * n_sys;
    std::size_t fwd_recv = rt_sz;
    buf_capacity_ = std::max(fwd_send, fwd_recv);
    CUDA_CHECK(cudaMalloc(&d_sendbuf_, sizeof(double) * buf_capacity_));
    CUDA_CHECK(cudaMalloc(&d_recvbuf_, sizeof(double) * buf_capacity_));

    // Merged 3x buffers for alltoall_forward_3 (A+C+D in one Alltoallv).
    // Mirrors PaScaL_TDMA_cuda.f90: BIGbuf size = 3 x normal buf.
    buf3_capacity_ = 3 * buf_capacity_;
    CUDA_CHECK(cudaMalloc(&d_sendbuf3_, sizeof(double) * buf3_capacity_));
    CUDA_CHECK(cudaMalloc(&d_recvbuf3_, sizeof(double) * buf3_capacity_));
    send_counts_3x_.resize(nprocs); send_displs_3x_.resize(nprocs);
    recv_counts_3x_.resize(nprocs); recv_displs_3x_.resize(nprocs);
    for (int p = 0; p < nprocs; ++p) {
        send_counts_3x_[p] = 3 * send_counts_fwd_[p];
        send_displs_3x_[p] = 3 * send_displs_fwd_[p];
        recv_counts_3x_[p] = 3 * recv_counts_fwd_[p];
        recv_displs_3x_[p] = 3 * recv_displs_fwd_[p];
    }

    // Cache max ns_rt for kernel grid sizing
    max_ns_rt_ = 0;
    for (int p = 0; p < nprocs; ++p)
        if (h_ns_rt_[p] > max_ns_rt_) max_ns_rt_ = h_ns_rt_[p];

    // ns_rt + col_offset + 4 displs arrays on device (kernels need them)
    CUDA_CHECK(cudaMalloc(&d_ns_rt_,           sizeof(int) * nprocs));
    CUDA_CHECK(cudaMalloc(&d_col_offset_,      sizeof(int) * nprocs));
    CUDA_CHECK(cudaMalloc(&d_send_displs_fwd_, sizeof(int) * nprocs));
    CUDA_CHECK(cudaMalloc(&d_recv_displs_fwd_, sizeof(int) * nprocs));
    CUDA_CHECK(cudaMalloc(&d_send_displs_bwd_, sizeof(int) * nprocs));
    CUDA_CHECK(cudaMalloc(&d_recv_displs_bwd_, sizeof(int) * nprocs));
    CUDA_CHECK(cudaMemcpy(d_ns_rt_,           h_ns_rt_.data(),         sizeof(int) * nprocs, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_offset_,      h_col_offset_.data(),    sizeof(int) * nprocs, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_send_displs_fwd_, send_displs_fwd_.data(), sizeof(int) * nprocs, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_recv_displs_fwd_, recv_displs_fwd_.data(), sizeof(int) * nprocs, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_send_displs_bwd_, send_displs_bwd_.data(), sizeof(int) * nprocs, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_recv_displs_bwd_, recv_displs_bwd_.data(), sizeof(int) * nprocs, cudaMemcpyHostToDevice));

    // Per-step timing events.  cudaEventBlockingSync makes cudaEventSynchronize
    // yield the CPU instead of spin-waiting — important when multiple ranks
    // share a node.
    for (int i = 0; i < 6; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&e_[i], cudaEventBlockingSync));
    }
}

// ===========================================================================
//  Destructor
// ===========================================================================
PaScaLTDMACUDA::~PaScaLTDMACUDA() {
    auto safe_free = [](double*& p) {
        if (p) { cudaFree(p); p = nullptr; }
    };
    safe_free(d_A_rd_); safe_free(d_B_rd_); safe_free(d_C_rd_); safe_free(d_D_rd_);
    safe_free(d_A_rt_); safe_free(d_B_rt_); safe_free(d_C_rt_); safe_free(d_D_rt_);
    safe_free(d_E_rt_); safe_free(d_E_loc_);
    safe_free(d_sendbuf_); safe_free(d_recvbuf_);
    safe_free(d_sendbuf3_); safe_free(d_recvbuf3_);
    auto free_int = [](int*& p) { if (p) { cudaFree(p); p = nullptr; } };
    free_int(d_ns_rt_);
    free_int(d_col_offset_);
    free_int(d_send_displs_fwd_);
    free_int(d_recv_displs_fwd_);
    free_int(d_send_displs_bwd_);
    free_int(d_recv_displs_bwd_);

    for (int i = 0; i < 6; ++i) {
        if (e_[i]) { cudaEventDestroy(e_[i]); e_[i] = nullptr; }
    }
}

// ===========================================================================
//  Lazy allocation of cyclic local workspace
// ===========================================================================
void PaScaLTDMACUDA::ensure_E_loc(int n_sys, int n_row) {
    std::size_t need = (std::size_t)n_row * (std::size_t)n_sys;
    if (need <= e_loc_capacity_) return;
    if (d_E_loc_) cudaFree(d_E_loc_);
    CUDA_CHECK(cudaMalloc(&d_E_loc_, sizeof(double) * need));
    e_loc_capacity_ = need;
}

// ===========================================================================
//  Merged forward: A_rd + C_rd + D_rd → A_rt + C_rt + D_rt (1 Alltoallv)
//  Fortran equivalent: pascal_a2av(BIGbuf_A, 3×size, BIGbufsubsize_A, ...)
// ===========================================================================
void PaScaLTDMACUDA::alltoall_forward_3(
    const double* d_A_rd, const double* d_C_rd, const double* d_D_rd,
    double* d_A_rt, double* d_C_rt, double* d_D_rt)
{
    NVTX_SCOPE("alltoall_fwd3");
    {
        NVTX_SCOPE("pack3_fwd");
        dim3 block(128, 1, 1);
        dim3 grid((max_ns_rt_ + block.x - 1) / block.x, N_ROW_RD, nprocs_);
        k_pack3_rd2send<<<grid, block>>>(d_sendbuf3_,
                                         d_A_rd, d_C_rd, d_D_rd,
                                         n_sys_,
                                         d_col_offset_, d_ns_rt_,
                                         d_send_displs_fwd_, nprocs_);
    }
    {
        NVTX_SCOPE("mpi_alltoallv_fwd3");
        CUDA_CHECK(cudaStreamSynchronize(0));
        double t0 = MPI_Wtime();
        MPI_Alltoallv(d_sendbuf3_, send_counts_3x_.data(), send_displs_3x_.data(),
                      MPI_DOUBLE,
                      d_recvbuf3_, recv_counts_3x_.data(), recv_displs_3x_.data(),
                      MPI_DOUBLE, comm_);
        last_comm_ms_ += (MPI_Wtime() - t0) * 1e3;
    }
    {
        NVTX_SCOPE("unpack3_fwd");
        dim3 block(128, 1, 1);
        dim3 grid((n_sys_rt_ + block.x - 1) / block.x, N_ROW_RD, nprocs_);
        k_unpack3_recv2rt<<<grid, block>>>(d_A_rt, d_C_rt, d_D_rt,
                                           d_recvbuf3_, n_sys_rt_, n_row_rt_,
                                           d_recv_displs_fwd_, nprocs_);
    }
}

// ===========================================================================
//  Backward all-to-all: [n_row_rt × n_sys_rt] rt  →  [2 × n_sys] rd
// ===========================================================================
void PaScaLTDMACUDA::alltoall_backward(const double* d_rt, double* d_rd) {
    NVTX_SCOPE("alltoall_bwd");
    // Pack: rt → sendbuf
    {
        NVTX_SCOPE("pack_bwd");
        dim3 block(128, 1, 1);
        dim3 grid((n_sys_rt_ + block.x - 1) / block.x, N_ROW_RD, nprocs_);
        k_pack_rt2send<<<grid, block>>>(d_sendbuf_, d_rt, n_sys_rt_,
                                        d_send_displs_bwd_, nprocs_);
    }

    {
        NVTX_SCOPE("mpi_alltoallv_bwd");
        CUDA_CHECK(cudaStreamSynchronize(0));
        double t0 = MPI_Wtime();
        MPI_Alltoallv(d_sendbuf_, send_counts_bwd_.data(), send_displs_bwd_.data(),
                      MPI_DOUBLE,
                      d_recvbuf_, recv_counts_bwd_.data(), recv_displs_bwd_.data(),
                      MPI_DOUBLE, comm_);
        last_comm_ms_ += (MPI_Wtime() - t0) * 1e3;
    }

    // Unpack: recvbuf → rd
    {
        NVTX_SCOPE("unpack_bwd");
        dim3 block(128, 1, 1);
        dim3 grid((max_ns_rt_ + block.x - 1) / block.x, N_ROW_RD, nprocs_);
        k_unpack_recv2rd<<<grid, block>>>(d_rd, d_recvbuf_, n_sys_,
                                          d_col_offset_, d_ns_rt_,
                                          d_recv_displs_bwd_, nprocs_);
    }
}

// ===========================================================================
//  solve()  — lean production path, no timing overhead
// ===========================================================================
void PaScaLTDMACUDA::solve(double* d_A, double* d_B, double* d_C, double* d_D,
                               int n_sys, int n_row) {
    if (nprocs_ == 1) {
        NVTX_SCOPE("tdma_local");
        tdma_many_cuda(d_A, d_B, d_C, d_D, n_sys, n_row, block_x_, block_y_);
        return;
    }

    NVTX_SCOPE("pascal_solve");
    pascal_tdma_modified_thomas_cuda(d_A, d_B, d_C, d_D,
                                     d_A_rd_, d_B_rd_, d_C_rd_, d_D_rd_,
                                     n_sys, n_row, block_x_, block_y_);
    alltoall_forward_3(d_A_rd_, d_C_rd_, d_D_rd_, d_A_rt_, d_C_rt_, d_D_rt_);
    tdma_many_cuda(d_A_rt_, d_B_rt_, d_C_rt_, d_D_rt_,
                   n_sys_rt_, n_row_rt_, block_x_, block_y_);
    alltoall_backward(d_D_rt_, d_D_rd_);
    pascal_tdma_update_solution_cuda(d_A, d_C, d_D, d_D_rd_,
                                     n_sys, n_row, block_x_, block_y_);
}

// ===========================================================================
//  solve_profile()  — same numerics + accurate pre/comm/post timing
//
//  Pack is absorbed into PRE; unpack is absorbed into POST.
//  The alltoall helpers are NOT called here so we can place cudaEvents
//  cleanly around every GPU kernel without spanning MPI idle time.
//
//  pre  [e_[0]→e_[1]] : modified_thomas_fwd + pack3_fwd
//  comm (MPI_Wtime)   : MPI_Alltoallv_fwd  +  MPI_Alltoallv_bwd → last_comm_ms_
//  post [e_[2]→e_[3]] : unpack3_fwd + reduced_tdma + pack_bwd
//       [e_[4]→e_[5]] : unpack_bwd  + update_solution
//
//  last_gpu_ms_  = ms_pre + ms_post_a + ms_post_b  (ALL GPU kernels, no MPI idle)
//  last_comm_ms_ = pure MPI_Alltoallv time (fwd + bwd)
// ===========================================================================
void PaScaLTDMACUDA::solve_profile(double* d_A, double* d_B, double* d_C, double* d_D,
                                       int n_sys, int n_row) {
    last_comm_ms_ = 0.0;
    last_gpu_ms_  = 0.0;

    if (nprocs_ == 1) {
        tdma_many_cuda(d_A, d_B, d_C, d_D, n_sys, n_row, block_x_, block_y_);
        return;
    }

    NVTX_SCOPE("pascal_solve_profile");
    dim3 blk(128, 1, 1);

    // === PRE: modified_thomas_fwd + pack3_fwd ===
    cudaEventRecord(e_[0]);
    pascal_tdma_modified_thomas_cuda(d_A, d_B, d_C, d_D,
                                     d_A_rd_, d_B_rd_, d_C_rd_, d_D_rd_,
                                     n_sys, n_row, block_x_, block_y_);
    {
        dim3 g((max_ns_rt_ + 127) / 128, N_ROW_RD, nprocs_);
        k_pack3_rd2send<<<g, blk>>>(d_sendbuf3_,
                                    d_A_rd_, d_C_rd_, d_D_rd_, n_sys_,
                                    d_col_offset_, d_ns_rt_,
                                    d_send_displs_fwd_, nprocs_);
    }
    cudaEventRecord(e_[1]);

    // === COMM fwd ===
    CUDA_CHECK(cudaStreamSynchronize(0));
    double t0 = MPI_Wtime();
    MPI_Alltoallv(d_sendbuf3_, send_counts_3x_.data(), send_displs_3x_.data(), MPI_DOUBLE,
                  d_recvbuf3_, recv_counts_3x_.data(), recv_displs_3x_.data(), MPI_DOUBLE, comm_);
    last_comm_ms_ += (MPI_Wtime() - t0) * 1e3;

    // === POST-a: unpack3_fwd + reduced_tdma + pack_bwd ===
    cudaEventRecord(e_[2]);
    {
        dim3 g((n_sys_rt_ + 127) / 128, N_ROW_RD, nprocs_);
        k_unpack3_recv2rt<<<g, blk>>>(d_A_rt_, d_C_rt_, d_D_rt_,
                                      d_recvbuf3_, n_sys_rt_, n_row_rt_,
                                      d_recv_displs_fwd_, nprocs_);
    }
    tdma_many_cuda(d_A_rt_, d_B_rt_, d_C_rt_, d_D_rt_,
                   n_sys_rt_, n_row_rt_, block_x_, block_y_);
    {
        dim3 g((n_sys_rt_ + 127) / 128, N_ROW_RD, nprocs_);
        k_pack_rt2send<<<g, blk>>>(d_sendbuf_, d_D_rt_, n_sys_rt_,
                                   d_send_displs_bwd_, nprocs_);
    }
    cudaEventRecord(e_[3]);

    // === COMM bwd ===
    CUDA_CHECK(cudaStreamSynchronize(0));
    t0 = MPI_Wtime();
    MPI_Alltoallv(d_sendbuf_, send_counts_bwd_.data(), send_displs_bwd_.data(), MPI_DOUBLE,
                  d_recvbuf_, recv_counts_bwd_.data(), recv_displs_bwd_.data(), MPI_DOUBLE, comm_);
    last_comm_ms_ += (MPI_Wtime() - t0) * 1e3;

    // === POST-b: unpack_bwd + update_solution ===
    cudaEventRecord(e_[4]);
    {
        dim3 g((max_ns_rt_ + 127) / 128, N_ROW_RD, nprocs_);
        k_unpack_recv2rd<<<g, blk>>>(d_D_rd_, d_recvbuf_, n_sys_,
                                     d_col_offset_, d_ns_rt_,
                                     d_recv_displs_bwd_, nprocs_);
    }
    pascal_tdma_update_solution_cuda(d_A, d_C, d_D, d_D_rd_,
                                     n_sys, n_row, block_x_, block_y_);
    cudaEventRecord(e_[5]);

    cudaEventSynchronize(e_[5]);
    float ms_pre = 0.0f, ms_post_a = 0.0f, ms_post_b = 0.0f;
    cudaEventElapsedTime(&ms_pre,    e_[0], e_[1]);
    cudaEventElapsedTime(&ms_post_a, e_[2], e_[3]);
    cudaEventElapsedTime(&ms_post_b, e_[4], e_[5]);
    last_gpu_ms_ = (double)(ms_pre + ms_post_a + ms_post_b);
}

// ===========================================================================
//  solve_cyclic()  — lean production path, no timing overhead
// ===========================================================================
void PaScaLTDMACUDA::solve_cyclic(double* d_A, double* d_B, double* d_C, double* d_D,
                                      int n_sys, int n_row) {
    if (nprocs_ == 1) {
        ensure_E_loc(n_sys, n_row);
        tdma_cyclic_many_cuda(d_A, d_B, d_C, d_D, d_E_loc_,
                              n_sys, n_row, block_x_, block_y_);
        return;
    }

    pascal_tdma_modified_thomas_cuda(d_A, d_B, d_C, d_D,
                                     d_A_rd_, d_B_rd_, d_C_rd_, d_D_rd_,
                                     n_sys, n_row, block_x_, block_y_);
    alltoall_forward_3(d_A_rd_, d_C_rd_, d_D_rd_, d_A_rt_, d_C_rt_, d_D_rt_);
    tdma_cyclic_many_cuda(d_A_rt_, d_B_rt_, d_C_rt_, d_D_rt_, d_E_rt_,
                          n_sys_rt_, n_row_rt_, block_x_, block_y_);
    alltoall_backward(d_D_rt_, d_D_rd_);
    pascal_tdma_update_solution_cuda(d_A, d_C, d_D, d_D_rd_,
                                     n_sys, n_row, block_x_, block_y_);
}

// ===========================================================================
//  solve_cyclic_profile()  — same numerics + accurate pre/comm/post timing
//  Same structure as solve_profile() but uses tdma_cyclic_many_cuda for the
//  reduced system.
// ===========================================================================
void PaScaLTDMACUDA::solve_cyclic_profile(double* d_A, double* d_B, double* d_C, double* d_D,
                                              int n_sys, int n_row) {
    last_comm_ms_ = 0.0;
    last_gpu_ms_  = 0.0;

    if (nprocs_ == 1) {
        ensure_E_loc(n_sys, n_row);
        tdma_cyclic_many_cuda(d_A, d_B, d_C, d_D, d_E_loc_,
                              n_sys, n_row, block_x_, block_y_);
        return;
    }

    dim3 blk(128, 1, 1);

    // === PRE: modified_thomas_fwd + pack3_fwd ===
    cudaEventRecord(e_[0]);
    pascal_tdma_modified_thomas_cuda(d_A, d_B, d_C, d_D,
                                     d_A_rd_, d_B_rd_, d_C_rd_, d_D_rd_,
                                     n_sys, n_row, block_x_, block_y_);
    {
        dim3 g((max_ns_rt_ + 127) / 128, N_ROW_RD, nprocs_);
        k_pack3_rd2send<<<g, blk>>>(d_sendbuf3_,
                                    d_A_rd_, d_C_rd_, d_D_rd_, n_sys_,
                                    d_col_offset_, d_ns_rt_,
                                    d_send_displs_fwd_, nprocs_);
    }
    cudaEventRecord(e_[1]);

    // === COMM fwd ===
    CUDA_CHECK(cudaStreamSynchronize(0));
    double t0 = MPI_Wtime();
    MPI_Alltoallv(d_sendbuf3_, send_counts_3x_.data(), send_displs_3x_.data(), MPI_DOUBLE,
                  d_recvbuf3_, recv_counts_3x_.data(), recv_displs_3x_.data(), MPI_DOUBLE, comm_);
    last_comm_ms_ += (MPI_Wtime() - t0) * 1e3;

    // === POST-a: unpack3_fwd + reduced_cyclic_tdma + pack_bwd ===
    cudaEventRecord(e_[2]);
    {
        dim3 g((n_sys_rt_ + 127) / 128, N_ROW_RD, nprocs_);
        k_unpack3_recv2rt<<<g, blk>>>(d_A_rt_, d_C_rt_, d_D_rt_,
                                      d_recvbuf3_, n_sys_rt_, n_row_rt_,
                                      d_recv_displs_fwd_, nprocs_);
    }
    tdma_cyclic_many_cuda(d_A_rt_, d_B_rt_, d_C_rt_, d_D_rt_, d_E_rt_,
                          n_sys_rt_, n_row_rt_, block_x_, block_y_);
    {
        dim3 g((n_sys_rt_ + 127) / 128, N_ROW_RD, nprocs_);
        k_pack_rt2send<<<g, blk>>>(d_sendbuf_, d_D_rt_, n_sys_rt_,
                                   d_send_displs_bwd_, nprocs_);
    }
    cudaEventRecord(e_[3]);

    // === COMM bwd ===
    CUDA_CHECK(cudaStreamSynchronize(0));
    t0 = MPI_Wtime();
    MPI_Alltoallv(d_sendbuf_, send_counts_bwd_.data(), send_displs_bwd_.data(), MPI_DOUBLE,
                  d_recvbuf_, recv_counts_bwd_.data(), recv_displs_bwd_.data(), MPI_DOUBLE, comm_);
    last_comm_ms_ += (MPI_Wtime() - t0) * 1e3;

    // === POST-b: unpack_bwd + update_solution ===
    cudaEventRecord(e_[4]);
    {
        dim3 g((max_ns_rt_ + 127) / 128, N_ROW_RD, nprocs_);
        k_unpack_recv2rd<<<g, blk>>>(d_D_rd_, d_recvbuf_, n_sys_,
                                     d_col_offset_, d_ns_rt_,
                                     d_recv_displs_bwd_, nprocs_);
    }
    pascal_tdma_update_solution_cuda(d_A, d_C, d_D, d_D_rd_,
                                     n_sys, n_row, block_x_, block_y_);
    cudaEventRecord(e_[5]);

    cudaEventSynchronize(e_[5]);
    float ms_pre = 0.0f, ms_post_a = 0.0f, ms_post_b = 0.0f;
    cudaEventElapsedTime(&ms_pre,    e_[0], e_[1]);
    cudaEventElapsedTime(&ms_post_a, e_[2], e_[3]);
    cudaEventElapsedTime(&ms_post_b, e_[4], e_[5]);
    last_gpu_ms_ = (double)(ms_pre + ms_post_a + ms_post_b);
}
