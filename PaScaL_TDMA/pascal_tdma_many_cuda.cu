#include "pascal_tdma_many_cuda.hpp"
#include "tdma_local_cuda.cuh"
#include "para_range.hpp"

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

namespace {

constexpr int N_ROW_RD = 2;  // reduced system has 2 rows per rank (top + bottom)

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

// ---------------------------------------------------------------------------
//  Pack:  [2 × n_sys] reduced buffer  →  contiguous sendbuf
//  For each rank p: writes 2 rows × ns_rt[p] cols starting at send_displs[p].
//  Within the per-rank block: row 0 of all cols, then row 1.
// ---------------------------------------------------------------------------
__global__ void pack_rd2send_kernel(double* __restrict__ sendbuf,
                                    const double* __restrict__ rd,
                                    int n_sys,
                                    const int* __restrict__ col_offset,
                                    const int* __restrict__ ns_rt,
                                    const int* __restrict__ send_displs,
                                    int nprocs) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y;            // 0 or 1
    int p = blockIdx.z;            // dest rank
    if (p >= nprocs || r >= N_ROW_RD) return;
    int ns_p = ns_rt[p];
    if (c >= ns_p) return;
    int src_col = col_offset[p] + c;
    sendbuf[send_displs[p] + r * ns_p + c] = rd[r * n_sys + src_col];
}

// ---------------------------------------------------------------------------
//  Unpack:  contiguous recvbuf  →  [n_row_rt × n_sys_rt] transposed buffer
//  Sender p's data occupies rows [2p, 2p+1] of rt; receive_displs[p] in recvbuf.
// ---------------------------------------------------------------------------
__global__ void unpack_recv2rt_kernel(double* __restrict__ rt,
                                      const double* __restrict__ recvbuf,
                                      int n_sys_rt, int n_row_rt,
                                      const int* __restrict__ recv_displs,
                                      int nprocs) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y;            // 0 or 1
    int p = blockIdx.z;            // sender rank
    if (p >= nprocs || r >= N_ROW_RD || c >= n_sys_rt) return;
    int dst_row = N_ROW_RD * p + r;
    rt[(std::size_t)dst_row * n_sys_rt + c]
        = recvbuf[recv_displs[p] + r * n_sys_rt + c];
}

// ---------------------------------------------------------------------------
//  Pack backward:  [n_row_rt × n_sys_rt] rt buffer  →  contiguous sendbuf
//  My rows for dest rank p are at rt[2p:2p+2], dest block at send_displs_bwd[p].
// ---------------------------------------------------------------------------
__global__ void pack_rt2send_kernel(double* __restrict__ sendbuf,
                                    const double* __restrict__ rt,
                                    int n_sys_rt,
                                    const int* __restrict__ send_displs_bwd,
                                    int nprocs) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y;
    int p = blockIdx.z;
    if (p >= nprocs || r >= N_ROW_RD || c >= n_sys_rt) return;
    int src_row = N_ROW_RD * p + r;
    sendbuf[send_displs_bwd[p] + r * n_sys_rt + c]
        = rt[(std::size_t)src_row * n_sys_rt + c];
}

// ---------------------------------------------------------------------------
//  Unpack backward:  contiguous recvbuf  →  [2 × n_sys] rd buffer
//  Sender p's data goes to columns [col_offset[p], col_offset[p]+ns_rt[p]).
// ---------------------------------------------------------------------------
__global__ void unpack_recv2rd_kernel(double* __restrict__ rd,
                                      const double* __restrict__ recvbuf,
                                      int n_sys,
                                      const int* __restrict__ col_offset,
                                      const int* __restrict__ ns_rt,
                                      const int* __restrict__ recv_displs_bwd,
                                      int nprocs) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y;
    int p = blockIdx.z;
    if (p >= nprocs || r >= N_ROW_RD) return;
    int ns_p = ns_rt[p];
    if (c >= ns_p) return;
    int dst_col = col_offset[p] + c;
    rd[r * n_sys + dst_col]
        = recvbuf[recv_displs_bwd[p] + r * ns_p + c];
}

} // namespace

// ===========================================================================
//  Constructor
// ===========================================================================
PaScaLTDMAManyCUDA::PaScaLTDMAManyCUDA(int n_sys, int myrank, int nprocs, MPI_Comm comm,
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
    cuda_fill(d_B_rt_, 1.0, rt_sz);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Pack/unpack scratch buffers. Forward sendbuf size = 2*n_sys; forward recvbuf
    // size = rt_sz. Same totals in reverse direction → reuse single pair sized to
    // the max of both (typically nearly equal).
    std::size_t fwd_send = (std::size_t)N_ROW_RD * n_sys;
    std::size_t fwd_recv = rt_sz;
    buf_capacity_ = std::max(fwd_send, fwd_recv);
    CUDA_CHECK(cudaMalloc(&d_sendbuf_, sizeof(double) * buf_capacity_));
    CUDA_CHECK(cudaMalloc(&d_recvbuf_, sizeof(double) * buf_capacity_));

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
PaScaLTDMAManyCUDA::~PaScaLTDMAManyCUDA() {
    auto safe_free = [](double*& p) {
        if (p) { cudaFree(p); p = nullptr; }
    };
    safe_free(d_A_rd_); safe_free(d_B_rd_); safe_free(d_C_rd_); safe_free(d_D_rd_);
    safe_free(d_A_rt_); safe_free(d_B_rt_); safe_free(d_C_rt_); safe_free(d_D_rt_);
    safe_free(d_E_rt_); safe_free(d_E_loc_);
    safe_free(d_sendbuf_); safe_free(d_recvbuf_);
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
void PaScaLTDMAManyCUDA::ensure_E_loc(int n_sys, int n_row) {
    std::size_t need = (std::size_t)n_row * (std::size_t)n_sys;
    if (need <= e_loc_capacity_) return;
    if (d_E_loc_) cudaFree(d_E_loc_);
    CUDA_CHECK(cudaMalloc(&d_E_loc_, sizeof(double) * need));
    e_loc_capacity_ = need;
}

// ===========================================================================
//  Forward all-to-all: [2 × n_sys] rd  →  [n_row_rt × n_sys_rt] rt
// ===========================================================================
void PaScaLTDMAManyCUDA::alltoall_forward(const double* d_rd, double* d_rt) {
    // Pack: rd → sendbuf
    {
        dim3 block(128, 1, 1);
        dim3 grid((max_ns_rt_ + block.x - 1) / block.x, N_ROW_RD, nprocs_);
        pack_rd2send_kernel<<<grid, block>>>(d_sendbuf_, d_rd, n_sys_,
                                             d_col_offset_, d_ns_rt_,
                                             d_send_displs_fwd_, nprocs_);
    }

    // Make pack visible to MPI before alltoall.  cudaStreamSynchronize on the
    // default stream is sufficient — matches PaScaL_TDMA_F.
    CUDA_CHECK(cudaStreamSynchronize(0));

    MPI_Alltoallv(d_sendbuf_, send_counts_fwd_.data(), send_displs_fwd_.data(),
                  MPI_DOUBLE,
                  d_recvbuf_, recv_counts_fwd_.data(), recv_displs_fwd_.data(),
                  MPI_DOUBLE, comm_);

    // Unpack: recvbuf → rt
    {
        dim3 block(128, 1, 1);
        dim3 grid((n_sys_rt_ + block.x - 1) / block.x, N_ROW_RD, nprocs_);
        unpack_recv2rt_kernel<<<grid, block>>>(d_rt, d_recvbuf_,
                                               n_sys_rt_, n_row_rt_,
                                               d_recv_displs_fwd_, nprocs_);
    }
}

// ===========================================================================
//  Backward all-to-all: [n_row_rt × n_sys_rt] rt  →  [2 × n_sys] rd
// ===========================================================================
void PaScaLTDMAManyCUDA::alltoall_backward(const double* d_rt, double* d_rd) {
    // Pack: rt → sendbuf
    {
        dim3 block(128, 1, 1);
        dim3 grid((n_sys_rt_ + block.x - 1) / block.x, N_ROW_RD, nprocs_);
        pack_rt2send_kernel<<<grid, block>>>(d_sendbuf_, d_rt, n_sys_rt_,
                                             d_send_displs_bwd_, nprocs_);
    }

    CUDA_CHECK(cudaStreamSynchronize(0));

    MPI_Alltoallv(d_sendbuf_, send_counts_bwd_.data(), send_displs_bwd_.data(),
                  MPI_DOUBLE,
                  d_recvbuf_, recv_counts_bwd_.data(), recv_displs_bwd_.data(),
                  MPI_DOUBLE, comm_);

    // Unpack: recvbuf → rd
    {
        dim3 block(128, 1, 1);
        dim3 grid((max_ns_rt_ + block.x - 1) / block.x, N_ROW_RD, nprocs_);
        unpack_recv2rd_kernel<<<grid, block>>>(d_rd, d_recvbuf_, n_sys_,
                                               d_col_offset_, d_ns_rt_,
                                               d_recv_displs_bwd_, nprocs_);
    }
}

// ===========================================================================
//  solve()
// ===========================================================================
void PaScaLTDMAManyCUDA::solve(double* d_A, double* d_B, double* d_C, double* d_D,
                               int n_sys, int n_row) {
    if (nprocs_ == 1) {
        tdma_many_cuda(d_A, d_B, d_C, d_D, n_sys, n_row, block_x_, block_y_);
        return;
    }

    pascal_tdma_modified_thomas_cuda(d_A, d_B, d_C, d_D,
                                     d_A_rd_, d_B_rd_, d_C_rd_, d_D_rd_,
                                     n_sys, n_row, block_x_, block_y_);
    alltoall_forward(d_A_rd_, d_A_rt_);
    alltoall_forward(d_C_rd_, d_C_rt_);
    alltoall_forward(d_D_rd_, d_D_rt_);
    tdma_many_cuda(d_A_rt_, d_B_rt_, d_C_rt_, d_D_rt_,
                   n_sys_rt_, n_row_rt_, block_x_, block_y_);
    alltoall_backward(d_D_rt_, d_D_rd_);
    pascal_tdma_update_solution_cuda(d_A, d_C, d_D, d_D_rd_,
                                     n_sys, n_row, block_x_, block_y_);
}

// ===========================================================================
//  solve_cyclic()
// ===========================================================================
void PaScaLTDMAManyCUDA::solve_cyclic(double* d_A, double* d_B, double* d_C, double* d_D,
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

    alltoall_forward(d_A_rd_, d_A_rt_);
    alltoall_forward(d_C_rd_, d_C_rt_);
    alltoall_forward(d_D_rd_, d_D_rt_);

    tdma_cyclic_many_cuda(d_A_rt_, d_B_rt_, d_C_rt_, d_D_rt_, d_E_rt_,
                          n_sys_rt_, n_row_rt_, block_x_, block_y_);

    alltoall_backward(d_D_rt_, d_D_rd_);

    pascal_tdma_update_solution_cuda(d_A, d_C, d_D, d_D_rd_,
                                     n_sys, n_row, block_x_, block_y_);
}
