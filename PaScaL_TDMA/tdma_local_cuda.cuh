#ifndef TDMA_LOCAL_CUDA_CUH
#define TDMA_LOCAL_CUDA_CUH

#include <cuda_runtime.h>

// Default block size for 1D thread layout (one thread per independent system).
#ifndef PASCAL_TDMA_BLOCK_X
#define PASCAL_TDMA_BLOCK_X 128
#endif

// ----------------------------------------------------------------------------
//  Local (single-rank) Thomas algorithm on device, multiple RHS systems.
//  Layout: row-major [n_row × n_sys]; row j starts at offset j*n_sys.
//  All pointers are device pointers.
// ----------------------------------------------------------------------------
void tdma_many_cuda(double* d_A, double* d_B, double* d_C, double* d_D,
                    int n_sys, int n_row, cudaStream_t stream = 0);

// Cyclic version. d_E is a workspace buffer of size n_row * n_sys (device).
void tdma_cyclic_many_cuda(double* d_A, double* d_B, double* d_C, double* d_D,
                           double* d_E,
                           int n_sys, int n_row, cudaStream_t stream = 0);

// ----------------------------------------------------------------------------
//  Modified Thomas algorithm — forward elimination producing the 2-row
//  reduced system at boundaries (rows 0 and n_row-1).
//  Inputs A,B,C,D are mutated in place; A_rd/B_rd/C_rd/D_rd hold the
//  reduced system [2 × n_sys].
// ----------------------------------------------------------------------------
void pascal_tdma_modified_thomas_cuda(double* d_A, double* d_B, double* d_C, double* d_D,
                                      double* d_A_rd, double* d_B_rd,
                                      double* d_C_rd, double* d_D_rd,
                                      int n_sys, int n_row,
                                      cudaStream_t stream = 0);

// ----------------------------------------------------------------------------
//  Final update: given D_rd[0,1] as the reduced solution at this rank's
//  two boundary rows, propagate back into D[1..n_row-2] using
//      D[j] -= A[j]*D_rd0 + C[j]*D_rd1
//  and copy boundaries D[0]=D_rd0, D[n_row-1]=D_rd1.
// ----------------------------------------------------------------------------
void pascal_tdma_update_solution_cuda(double* d_A, double* d_C, double* d_D,
                                      const double* d_D_rd,
                                      int n_sys, int n_row,
                                      cudaStream_t stream = 0);

#endif // TDMA_LOCAL_CUDA_CUH
