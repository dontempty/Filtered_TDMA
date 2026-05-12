#ifndef PASCAL_TDMA_MANY_CUDA_HPP
#define PASCAL_TDMA_MANY_CUDA_HPP

#include <mpi.h>
#include <vector>

/// CUDA / GPU version of PaScaLTDMAMany.
///
/// Mirrors the CPU class (same algorithm, same MPI alltoall transpose pattern),
/// but the modified-Thomas / Thomas / update kernels run on device and the
/// reduced-system buffers live in GPU memory. All input pointers passed to
/// solve() must be device pointers, row-major [n_row × n_sys].
///
/// MPI calls operate directly on device pointers — a CUDA-aware MPI build is
/// required (matches the Fortran reference implementation).
class PaScaLTDMAManyCUDA {
public:
    PaScaLTDMAManyCUDA(int n_sys, int myrank, int nprocs, MPI_Comm comm);
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

private:
    // Lazily (re)allocate d_E_loc_ for the nprocs==1 cyclic path.
    void ensure_E_loc(int n_sys, int n_row);

    MPI_Comm comm_;
    int      nprocs_;
    int      n_sys_;       // fixed at construction; matches DDT
    int      n_sys_rt_;    // systems owned by this rank after transpose
    int      n_row_rt_;    // rows of reduced system after transpose (= 2*nprocs)

    // MPI derived datatypes (identical layout to CPU PaScaLTDMAMany)
    std::vector<MPI_Datatype> ddtype_Fs_;
    std::vector<MPI_Datatype> ddtype_Bs_;
    std::vector<int> count_send_, displ_send_;
    std::vector<int> count_recv_, displ_recv_;

    // Device buffers (raw cudaMalloc, freed in dtor)
    double* d_A_rd_ = nullptr;  // [2 × n_sys]
    double* d_B_rd_ = nullptr;
    double* d_C_rd_ = nullptr;
    double* d_D_rd_ = nullptr;

    double* d_A_rt_ = nullptr;  // [n_row_rt × n_sys_rt]
    double* d_B_rt_ = nullptr;  // initialized to 1.0
    double* d_C_rt_ = nullptr;
    double* d_D_rt_ = nullptr;
    double* d_E_rt_ = nullptr;  // workspace for cyclic local solve on reduced system

    // Workspace for nprocs==1 cyclic path (allocated lazily on first cyclic call).
    double* d_E_loc_ = nullptr;
    std::size_t e_loc_capacity_ = 0;
};

#endif // PASCAL_TDMA_MANY_CUDA_HPP
