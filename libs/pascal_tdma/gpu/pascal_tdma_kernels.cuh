#ifndef PASCAL_TDMA_KERNELS_CUH
#define PASCAL_TDMA_KERNELS_CUH

// Private CUDA kernel definitions shared between pascal_tdma_many_cuda.cu and
// any future profiling TU. Each TU gets its own static copy; no -rdc required.

#include <cuda_runtime.h>
#include <cstddef>

static constexpr int N_ROW_RD = 2;  // reduced system: 2 rows per rank (top + bottom)

static __global__ void k_fill(double* p, double v, std::size_t n) {
    std::size_t idx = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) p[idx] = v;
}

static inline void pascal_cuda_fill(double* p, double v, std::size_t n) {
    if (n == 0) return;
    int block = 256;
    int grid  = (int)((n + block - 1) / block);
    k_fill<<<grid, block>>>(p, v, n);
}

// ---------------------------------------------------------------------------
//  Forward pack:  [2 × n_sys] reduced buffer  →  contiguous sendbuf
//  For dest rank p: writes 2 rows × ns_rt[p] cols at send_displs[p].
//  Layout per rank block: row 0 of all cols, then row 1.
// ---------------------------------------------------------------------------
static __global__ void k_pack_rd2send(double* __restrict__ sendbuf,
                                      const double* __restrict__ rd,
                                      int n_sys,
                                      const int* __restrict__ col_offset,
                                      const int* __restrict__ ns_rt,
                                      const int* __restrict__ send_displs,
                                      int nprocs) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y;   // 0 or 1
    int p = blockIdx.z;   // dest rank
    if (p >= nprocs || r >= N_ROW_RD) return;
    int ns_p = ns_rt[p];
    if (c >= ns_p) return;
    int src_col = col_offset[p] + c;
    sendbuf[send_displs[p] + r * ns_p + c] = rd[r * n_sys + src_col];
}

// ---------------------------------------------------------------------------
//  Forward unpack:  contiguous recvbuf  →  [n_row_rt × n_sys_rt] rt buffer
//  Sender p's data occupies rows [2p, 2p+1] of rt.
// ---------------------------------------------------------------------------
static __global__ void k_unpack_recv2rt(double* __restrict__ rt,
                                        const double* __restrict__ recvbuf,
                                        int n_sys_rt, int n_row_rt,
                                        const int* __restrict__ recv_displs,
                                        int nprocs) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y;   // 0 or 1
    int p = blockIdx.z;   // sender rank
    if (p >= nprocs || r >= N_ROW_RD || c >= n_sys_rt) return;
    int dst_row = N_ROW_RD * p + r;
    rt[(std::size_t)dst_row * n_sys_rt + c]
        = recvbuf[recv_displs[p] + r * n_sys_rt + c];
}

// ---------------------------------------------------------------------------
//  Backward pack:  [n_row_rt × n_sys_rt] rt buffer  →  contiguous sendbuf
// ---------------------------------------------------------------------------
static __global__ void k_pack_rt2send(double* __restrict__ sendbuf,
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
//  Backward unpack:  contiguous recvbuf  →  [2 × n_sys] rd buffer
// ---------------------------------------------------------------------------
static __global__ void k_unpack_recv2rd(double* __restrict__ rd,
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

// ---------------------------------------------------------------------------
//  Merged forward pack: A_rd + C_rd + D_rd → single 3× send buffer.
//  For dest rank p: [A_block | C_block | D_block], each N_ROW_RD × ns_rt[p].
//  Displacement in 3× buffer = 3 × send_displs_fwd[p].
// ---------------------------------------------------------------------------
static __global__ void k_pack3_rd2send(double* __restrict__ send3,
                                       const double* __restrict__ A_rd,
                                       const double* __restrict__ C_rd,
                                       const double* __restrict__ D_rd,
                                       int n_sys,
                                       const int* __restrict__ col_offset,
                                       const int* __restrict__ ns_rt,
                                       const int* __restrict__ send_displs,
                                       int nprocs) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y;
    int p = blockIdx.z;
    if (p >= nprocs || r >= N_ROW_RD) return;
    int ns_p = ns_rt[p];
    if (c >= ns_p) return;
    int src  = col_offset[p] + c;
    int base = 3 * send_displs[p];
    int cnt  = N_ROW_RD * ns_p;
    int off  = r * ns_p + c;
    send3[base + off]           = A_rd[r * n_sys + src];
    send3[base + cnt   + off]   = C_rd[r * n_sys + src];
    send3[base + 2*cnt + off]   = D_rd[r * n_sys + src];
}

// ---------------------------------------------------------------------------
//  Merged forward unpack: 3× recv buffer → A_rt + C_rt + D_rt.
// ---------------------------------------------------------------------------
static __global__ void k_unpack3_recv2rt(double* __restrict__ A_rt,
                                         double* __restrict__ C_rt,
                                         double* __restrict__ D_rt,
                                         const double* __restrict__ recv3,
                                         int n_sys_rt, int n_row_rt,
                                         const int* __restrict__ recv_displs,
                                         int nprocs) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y;
    int p = blockIdx.z;
    if (p >= nprocs || r >= N_ROW_RD || c >= n_sys_rt) return;
    int dst_row = N_ROW_RD * p + r;
    int base = 3 * recv_displs[p];
    int cnt  = N_ROW_RD * n_sys_rt;
    int off  = r * n_sys_rt + c;
    std::size_t dst = (std::size_t)dst_row * n_sys_rt + c;
    A_rt[dst] = recv3[base + off];
    C_rt[dst] = recv3[base + cnt   + off];
    D_rt[dst] = recv3[base + 2*cnt + off];
}

#endif // PASCAL_TDMA_KERNELS_CUH
