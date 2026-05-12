#ifndef FILTERED_TDMA_CUDA_HPP
#define FILTERED_TDMA_CUDA_HPP

#include <mpi.h>
#include <vector>

/// CUDA / GPU version of FilteredTDMA.
///
/// Mirrors the CPU class (same DistD2 algorithm, same MPI subarray-based
/// boundary-row exchange), but every per-row sweep over `n_sys` runs as a
/// CUDA kernel and all working buffers (`A_rho`, `C_rho`, the 4 receive /
/// send rows of size `n_sys`) live in GPU memory.
///
/// All input pointers passed to `solve_filtered_v1/v2()` must be **device
/// pointers**, row-major `[n_row × n_sys]`. The MPI Isend/Irecv calls use
/// these device pointers directly — a CUDA-aware MPI build is required
/// (matches the `PaScaLTDMAManyCUDA` requirement on the same partition).
class FilteredTDMACUDA {
public:
    FilteredTDMACUDA(int n_sys, int n_row,
                     int myrank, int nprocs, MPI_Comm comm,
                     int left_rank, int right_rank,
                     double eps_constant);

    ~FilteredTDMACUDA();

    FilteredTDMACUDA(const FilteredTDMACUDA&)            = delete;
    FilteredTDMACUDA& operator=(const FilteredTDMACUDA&) = delete;

    /// Filtered_TDMA (v1). Device pointers required. `d_D` overwritten on
    /// return with the solution.
    void solve_filtered_v1(double* d_A, double* d_B, double* d_C, double* d_D);

    /// Filtered_TDMA (v2). Device pointers required.
    void solve_filtered_v2(double* d_A, double* d_B, double* d_C, double* d_D);

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
    /// Estimate cutoff index J from worst-case bounds (matches CPU exactly).
    /// `D0_dev`, `DN_dev` are pointers to row 0 / row n_row-1 of D on device.
    int cal_J_v1(const double* D0_dev, const double* DN_dev);
    int cal_J_v2(double D0, double DN);

    // --- MPI plan data ---
    MPI_Comm     comm_;
    int          n_sys_, n_row_, nprocs_;
    int          left_rank_, right_rank_;

    MPI_Datatype row0_type_ = MPI_DATATYPE_NULL;   // subarray type for row j=0
    MPI_Datatype rowN_type_ = MPI_DATATYPE_NULL;   // subarray type for row j=n_row-1

    // Device receive / send buffers for neighbor boundary data
    double* d_C_left_recv_   = nullptr;
    double* d_D_left_recv_   = nullptr;
    double* d_D_right_send_  = nullptr;
    double* d_D_right_recv_  = nullptr;

    // Per-row spectral radius estimates (device)
    double* d_A_rho_ = nullptr;
    double* d_C_rho_ = nullptr;

    // Host mirror of A_rho/C_rho (filled by cal_J via cudaMemcpy) — small,
    // (n_row+1) doubles, kept here to avoid per-call host alloc.
    std::vector<double> h_A_rho_;
    std::vector<double> h_C_rho_;

    // Device scratch for {max|D0|, max|DN|} reductions (2 doubles, one float
    // for v1 — mirrored back to host inside cal_J_v1).
    double* d_maxD_ = nullptr;     // [2]

    double eps_;
};

#endif // FILTERED_TDMA_CUDA_HPP
