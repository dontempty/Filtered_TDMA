#ifndef PASCAL_TDMA_MANY_CUDA_HPP
#define PASCAL_TDMA_MANY_CUDA_HPP

#include <mpi.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <vector>

/// CUDA / GPU version of PaScaLTDMAMany.
///
/// Communication pattern mirrors PaScaL_TDMA_F/src/pascal_tdma_cuda.f90 :
///   - explicit device-side pack into a contiguous send buffer,
///   - cudaStreamSynchronize + MPI_Alltoallv on the contiguous device pointer
///     (CUDA-aware MPI fast path),
///   - explicit device-side unpack into the transposed reduced-system buffer.
/// This avoids the OpenMPI/UCX fallback that subarray derived datatypes on
/// device pointers historically trigger (per-element D2H staging).
class PaScaLTDMAManyCUDA {
public:
    PaScaLTDMAManyCUDA(int n_sys, int myrank, int nprocs, MPI_Comm comm,
                       int block_x = 128, int block_y = 1);
    ~PaScaLTDMAManyCUDA();

    PaScaLTDMAManyCUDA(const PaScaLTDMAManyCUDA&)            = delete;
    PaScaLTDMAManyCUDA& operator=(const PaScaLTDMAManyCUDA&) = delete;

    /// Non-cyclic solve. d_A,d_B,d_C,d_D are device pointers; d_D is overwritten.
    void solve(double* d_A, double* d_B, double* d_C, double* d_D,
               int n_sys, int n_row);

    /// Cyclic variant.
    void solve_cyclic(double* d_A, double* d_B, double* d_C, double* d_D,
                      int n_sys, int n_row);

    int n_sys_rt() const { return n_sys_rt_; }
    int n_row_rt() const { return n_row_rt_; }
    int nprocs()   const { return nprocs_; }

    /// Sub-phase wall times (ms) measured on the GPU for the most recent
    /// non-cyclic solve() call (cyclic variant currently does not populate).
    /// Layout matches `acc_` semantics:
    ///   [0] mod_thomas_fwd
    ///   [1] alltoall_forward  (sum of A, C, D — three calls)
    ///   [2] tdma_many on reduced system
    ///   [3] alltoall_backward (single — D only)
    ///   [4] update_solution
    /// When nprocs==1 (no comm path) all entries are 0.
    const double* last_step_times_ms() const { return last_ms_; }

private:
    void ensure_E_loc(int n_sys, int n_row);

    // One direction of the all-to-all: pack [n_row_rd × n_sys] reduced buffer
    // into the contiguous sendbuf, exchange, unpack into the [n_row_rt × n_sys_rt]
    // transposed buffer.
    void alltoall_forward(const double* d_rd, double* d_rt);
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

    // Pack / unpack scratch (device).  Sized to max(2*n_sys, n_row_rt*n_sys_rt).
    double* d_sendbuf_ = nullptr;
    double* d_recvbuf_ = nullptr;
    std::size_t buf_capacity_ = 0;

    // Workspace for nprocs==1 cyclic path (allocated lazily on first cyclic call).
    double* d_E_loc_ = nullptr;
    std::size_t e_loc_capacity_ = 0;

    // ===== Per-step timing (diagnostic) =====
    // acc_[0] = modified_thomas
    // acc_[1] = alltoall_forward × 3 (A, C, D)
    // acc_[2] = local tdma_many on reduced system
    // acc_[3] = alltoall_backward × 1 (D)
    // acc_[4] = update_solution
    // Measured via CUDA events on the default stream — captures actual GPU
    // execution time without including host-side cudaDeviceSynchronize spin.
    double      acc_[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    int         call_count_ = 0;
    cudaEvent_t e_[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

    // Sub-phase times of the most recent solve() call (ms).
    // Indices: [0]=mod_thomas, [1]=alltoall_fwd_total, [2]=reduced_tdma,
    //          [3]=alltoall_bwd, [4]=update_sol.
    double      last_ms_[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
};

#endif // PASCAL_TDMA_MANY_CUDA_HPP
