#ifndef PASCAL_TDMA_CUDA_HPP
#define PASCAL_TDMA_CUDA_HPP

#include <mpi.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <vector>

/// CUDA / GPU version of PaScaLTDMA.
///
/// Communication pattern mirrors PaScaL_TDMA_F/src/pascal_tdma_cuda.f90 :
///   - explicit device-side pack into a contiguous send buffer,
///   - cudaStreamSynchronize + MPI_Alltoallv on the contiguous device pointer
///     (CUDA-aware MPI fast path),
///   - explicit device-side unpack into the transposed reduced-system buffer.
/// This avoids the OpenMPI/UCX fallback that subarray derived datatypes on
/// device pointers historically trigger (per-element D2H staging).
class PaScaLTDMACUDA {
public:
    PaScaLTDMACUDA(int n_sys, int myrank, int nprocs, MPI_Comm comm,
                       int block_x = 128, int block_y = 1);
    ~PaScaLTDMACUDA();

    PaScaLTDMACUDA(const PaScaLTDMACUDA&)            = delete;
    PaScaLTDMACUDA& operator=(const PaScaLTDMACUDA&) = delete;

    /// Non-cyclic solve (lean production path — no timing overhead).
    void solve(double* d_A, double* d_B, double* d_C, double* d_D,
               int n_sys, int n_row);

    /// Cyclic variant (lean production path).
    void solve_cyclic(double* d_A, double* d_B, double* d_C, double* d_D,
                      int n_sys, int n_row);

    /// Profile variants: same numerics, adds pre/comm/post timing so
    /// last_comm_ms()/last_gpu_ms() return meaningful values.
    /// Use these in benchmark paths (e.g. heat_gpu); channel_gpu uses the lean variants.
    void solve_profile(double* d_A, double* d_B, double* d_C, double* d_D,
                       int n_sys, int n_row);
    void solve_cyclic_profile(double* d_A, double* d_B, double* d_C, double* d_D,
                              int n_sys, int n_row);

    int n_sys_rt() const { return n_sys_rt_; }
    int n_row_rt() const { return n_row_rt_; }
    int nprocs()   const { return nprocs_; }

    /// Pure MPI wall-clock time (ms) for the most recent solve_profile call.
    /// Accumulated via MPI_Wtime in alltoall_forward_3 + alltoall_backward.
    /// 0 when nprocs==1 or when solve() (non-profile) was called last.
    double last_comm_ms() const { return last_comm_ms_; }

    /// GPU kernel execution time (ms) for the most recent solve_profile call.
    /// Sum of modified_thomas_fwd + reduced_tdma + update_solution (pre+post).
    /// Measured via CUDA events; excludes MPI idle and pack/unpack kernels.
    /// 0 when nprocs==1 or when solve() (non-profile) was called last.
    double last_gpu_ms()  const { return last_gpu_ms_; }

private:
    void ensure_E_loc(int n_sys, int n_row);

    // Merged forward all-to-all: pack A_rd, C_rd, D_rd into a single 3× buffer
    // and perform ONE MPI_Alltoallv (vs the original 3 separate calls).
    // Mirrors the PaScaL_TDMA_cuda Fortran BIGbuf_A approach.
    void alltoall_forward_3(const double* d_A_rd, const double* d_C_rd, const double* d_D_rd,
                            double* d_A_rt, double* d_C_rt, double* d_D_rt);
    // Reverse: pack [n_row_rt × n_sys_rt], exchange, unpack into [n_row_rd × n_sys].
    void alltoall_backward(const double* d_rt, double* d_rd);

    MPI_Comm comm_;
    int      myrank_;
    int      nprocs_;
    int      n_sys_;       // fixed at construction; total columns of the reduced system
    int      n_sys_rt_;    // systems owned by this rank after transpose
    int      n_row_rt_;    // rows of reduced system after transpose (= 2*nprocs)

    // PaScaL TDMA kernel thread-block dimensions (passed at construction;
    // mirrors Fortran reference's thread_in_x_pascal / thread_in_y_pascal).
    int      block_x_;
    int      block_y_;

    // Per-rank column counts and offsets (host + device copies; device used by kernels)
    std::vector<int> h_ns_rt_;        // h_ns_rt_[p] = rank p's column count (after split of n_sys)
    std::vector<int> h_col_offset_;   // h_col_offset_[p] = exclusive prefix sum of h_ns_rt_
    int              max_ns_rt_ = 0;  // max over ranks (kernel grid bound)
    int*             d_ns_rt_      = nullptr;   // device copy
    int*             d_col_offset_ = nullptr;   // device copy

    // Pre-computed MPI_Alltoallv parameters (in MPI_DOUBLE units).
    // "_fwd": sending reduced-rows (rd → rt), "_bwd": sending solution back (rt → rd).
    std::vector<int> send_counts_fwd_, send_displs_fwd_;
    std::vector<int> recv_counts_fwd_, recv_displs_fwd_;
    std::vector<int> send_counts_bwd_, send_displs_bwd_;
    std::vector<int> recv_counts_bwd_, recv_displs_bwd_;

    // Device copies of the *_displs arrays (used by pack/unpack kernels)
    int* d_send_displs_fwd_ = nullptr;
    int* d_recv_displs_fwd_ = nullptr;
    int* d_send_displs_bwd_ = nullptr;
    int* d_recv_displs_bwd_ = nullptr;

    // Device buffers (raw cudaMalloc, freed in dtor)
    double* d_A_rd_ = nullptr;  // [2 × n_sys]
    double* d_B_rd_ = nullptr;
    double* d_C_rd_ = nullptr;
    double* d_D_rd_ = nullptr;

    double* d_A_rt_ = nullptr;  // [n_row_rt × n_sys_rt]
    double* d_B_rt_ = nullptr;  // initialized to 1.0 (b_rt is identity diagonal)
    double* d_C_rt_ = nullptr;
    double* d_D_rt_ = nullptr;
    double* d_E_rt_ = nullptr;  // workspace for cyclic local solve on reduced system

    // Pack / unpack scratch for the individual (legacy) alltoall_forward path.
    // (kept in dtor for safe free; not used by alltoall_forward_3)
    double* d_sendbuf_ = nullptr;
    double* d_recvbuf_ = nullptr;
    std::size_t buf_capacity_ = 0;

    // Merged 3× buffers for alltoall_forward_3 (A+C+D in one shot).
    // Mirrors PaScaL_TDMA_cuda.f90 BIGbuf_A / BIGbuf_B.
    double* d_sendbuf3_ = nullptr;
    double* d_recvbuf3_ = nullptr;
    std::size_t buf3_capacity_ = 0;
    std::vector<int> send_counts_3x_, send_displs_3x_;
    std::vector<int> recv_counts_3x_, recv_displs_3x_;

    // Workspace for nprocs==1 cyclic path (allocated lazily on first cyclic call).
    double* d_E_loc_ = nullptr;
    std::size_t e_loc_capacity_ = 0;

    // ===== Per-step profile timing (solve_profile / solve_cyclic_profile only) =====
    // 6 events: [0,1] bracket mod_thomas_fwd (pre),
    //           [2,3] bracket reduced_tdma (post-a),
    //           [4,5] bracket update_solution (post-b).
    // last_comm_ms_ is accumulated inside alltoall_forward_3 + alltoall_backward
    // via MPI_Wtime (captures pure MPI_Alltoallv time, forward + backward).
    cudaEvent_t e_[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    double      last_comm_ms_ = 0.0;
    double      last_gpu_ms_  = 0.0;
};

#endif // PASCAL_TDMA_CUDA_HPP
