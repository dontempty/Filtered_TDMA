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
//  block_x × block_y = threads per block (one thread per system).
// ----------------------------------------------------------------------------
void tdma_many_cuda(double* d_A, double* d_B, double* d_C, double* d_D,
                    int n_sys, int n_row,
                    int block_x = 128, int block_y = 1,
                    cudaStream_t stream = 0);

void tdma_cyclic_many_cuda(double* d_A, double* d_B, double* d_C, double* d_D,
                           double* d_E,
                           int n_sys, int n_row,
                           int block_x = 128, int block_y = 1,
                           cudaStream_t stream = 0);

// ----------------------------------------------------------------------------
//  Modified Thomas algorithm — forward elimination producing the 2-row
//  reduced system at boundaries (rows 0 and n_row-1).
// ----------------------------------------------------------------------------
void pascal_tdma_modified_thomas_cuda(double* d_A, double* d_B, double* d_C, double* d_D,
                                      double* d_A_rd, double* d_B_rd,
                                      double* d_C_rd, double* d_D_rd,
                                      int n_sys, int n_row,
                                      int block_x = 128, int block_y = 1,
                                      cudaStream_t stream = 0);

// ----------------------------------------------------------------------------
//  Final update.
// ----------------------------------------------------------------------------
void pascal_tdma_update_solution_cuda(double* d_A, double* d_C, double* d_D,
                                      const double* d_D_rd,
                                      int n_sys, int n_row,
                                      int block_x = 128, int block_y = 1,
                                      cudaStream_t stream = 0);

#endif // TDMA_LOCAL_CUDA_CUH
