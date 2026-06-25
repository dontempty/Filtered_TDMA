#ifndef FILTERED_TDMA_CUDA_HPP
#define FILTERED_TDMA_CUDA_HPP

#include <mpi.h>
#include <cuda_runtime.h>

/// CUDA implementation of the Filtered TDMA solver.
///
/// Solves a distributed tridiagonal system split across `nprocs` ranks along
/// the row dimension. Each rank holds `n_row` rows of `n_sys` independent
/// systems. All matrix/RHS arrays are device pointers (row-major, n_row×n_sys).
///
/// Communication uses CUDA-aware MPI (P2P Isend/Irecv with neighbor ranks).
/// Two packed 2×n_sys buffers are exchanged per solve, one in each direction.
///
/// v1: standard forward/backward sweep (all rows updated).
/// v2: filtered sweep — computes cutoff J before the forward pass and skips
///     A updates for rows beyond J, reducing compute when rho is small.
class FilteredTDMACUDA {
public:
    FilteredTDMACUDA(int n_sys, int n_row,
                     int myrank, int nprocs, MPI_Comm comm,
                     int left_rank, int right_rank,
                     double eps_constant);

    ~FilteredTDMACUDA();

    FilteredTDMACUDA(const FilteredTDMACUDA&)            = delete;
    FilteredTDMACUDA& operator=(const FilteredTDMACUDA&) = delete;

    void solve_filtered_v1(double* d_A, double* d_B, double* d_C, double* d_D);
    void solve_filtered_v2(double* d_A, double* d_B, double* d_C, double* d_D);

    /// Periodic variants: rank 0's left neighbor is the last rank, giving every
    /// rank real interface blocks on both sides. The multi-rank body is identical
    /// to the non-periodic solve; only the nprocs==1 fallback differs.
    void solve_cycl_filtered_v1(double* d_A, double* d_B, double* d_C, double* d_D);
    void solve_cycl_filtered_v2(double* d_A, double* d_B, double* d_C, double* d_D);

    int n_sys()  const { return n_sys_; }
    int n_row()  const { return n_row_; }
    int nprocs() const { return nprocs_; }

    /// Pointers to the per-row spectral-radius arrays on the device. Caller
    /// is responsible for filling these via `set_rho_device()` before each
    /// solve. Size of each: `n_row + 1` doubles (the trailing slot stays 0).
    double* d_A_rho() { return d_A_rho_; }
    double* d_C_rho() { return d_C_rho_; }

    /// Refresh eps from a new eps_constant (e.g. the current dt).
    /// eps_ = eps_constant / (n_row * nprocs)^2.
    void set_eps_constant(double eps_constant) {
        const double N = static_cast<double>(n_row_ * nprocs_);
        eps_ = eps_constant / (N * N);
    }

    /// Compute and store per-row A_rho/C_rho on the device from the current
    /// matrix entries. The CPU `TdmaBackend` calls the analogous host-side
    /// helper before each solve; this is the device counterpart used by
    /// the GPU dispatcher.
    void set_rho_device(const double* d_A, const double* d_B, const double* d_C);

private:
    // Estimate cutoff J from boundary rows D[0,:] and D[n_row-1,:] using the
    // v2 formula. Result written to d_J_ on device; also returns J via D2H copy.
    int cal_J_rhs_bound(const double* d_D);

    // Shared solve bodies for both the cyclic and non-cyclic public entry points.
    // Cyclic-ness is encoded in comm_ (periodic communicator → both boundaries active).
    void v1_multirank(double* d_A, double* d_B, double* d_C, double* d_D);
    void v2_multirank(double* d_A, double* d_B, double* d_C, double* d_D);

    // --- MPI plan data ---
    MPI_Comm     comm_;
    int          n_sys_, n_row_, nprocs_;
    int          left_rank_, right_rank_;

    // Packed 2-message exchange buffers (each 2*n_sys doubles).
    // send_right: [C_lastrow | D_lastrow] → right neighbor (tags 1)
    // send_left : [A_row0   | D_row0  ] → left  neighbor (tag 2)
    // recv_left : [C | D] from left  (C_left = recv_left, D_left = recv_left+n_sys)
    // recv_right: [A | D] from right (A_right= recv_right, D0_right= recv_right+n_sys)
    double* d_send_right_ = nullptr;
    double* d_send_left_  = nullptr;
    double* d_recv_left_  = nullptr;
    double* d_recv_right_ = nullptr;

    // Per-row spectral radius estimates (device)
    double* d_A_rho_ = nullptr;
    double* d_C_rho_ = nullptr;
    // Device scratch for cal_J kernel result (single int).
    int*    d_J_     = nullptr;
    // Lazily-allocated [n_row × n_sys] workspace for the nprocs==1 cyclic
    // (Sherman-Morrison) local solve. Unused by the non-cyclic path.
    double* d_E_     = nullptr;

    double eps_;

    // --- per-solve timing ---
    double last_comm_ms_ = 0.0;  // pure MPI time (after GPU sync)
    double last_gpu_ms_  = 0.0;  // GPU kernel time (CUDA events)

    // 4 events to bracket pre-comm and post-comm kernel groups.
    // Initialized in ctor, destroyed in dtor.
    cudaEvent_t e_gpu_start_     = nullptr;  // before first kernel
    cudaEvent_t e_gpu_pre_end_   = nullptr;  // after last pre-comm kernel
    cudaEvent_t e_gpu_post_start_= nullptr;  // before k_solve_both
    cudaEvent_t e_gpu_post_end_  = nullptr;  // after k_final_pass

public:
    /// Pure MPI wall-clock time (ms) for the most recent solve call.
    /// Measured AFTER cudaStreamSynchronize — excludes GPU sync wait.
    double last_comm_ms() const { return last_comm_ms_; }

    /// GPU kernel execution time (ms) for the most recent solve call.
    /// Sum of pre-comm kernels (fwd, bwd+pack, pack_lastrow/row0) and
    /// post-comm kernels (solve_both, cal_J, final_pass).
    double last_gpu_ms()  const { return last_gpu_ms_; }
};

#endif // FILTERED_TDMA_CUDA_HPP
