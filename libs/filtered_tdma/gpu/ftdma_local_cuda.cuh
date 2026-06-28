#ifndef FTDMA_LOCAL_CUDA_CUH
#define FTDMA_LOCAL_CUDA_CUH

#include <cuda_runtime.h>

// Local (single-rank) GPU TDMA kernels for the filtered_tdma library.
// These mirror the equivalent functions in pascal_tdma/tdma_local_cuda but
// are compiled independently (no cross-library symbol dependency).
// Naming prefix "ftdma_" avoids link-time conflicts when both libraries are
// linked into the same executable.

void ftdma_many_cuda(double* d_A, double* d_B, double* d_C, double* d_D,
                     int n_sys, int n_row,
                     int block_x = 128, int block_y = 1,
                     cudaStream_t stream = 0);

void ftdma_cyclic_many_cuda(double* d_A, double* d_B, double* d_C, double* d_D,
                            double* d_E,
                            int n_sys, int n_row,
                            int block_x = 128, int block_y = 1,
                            cudaStream_t stream = 0);

#endif // FTDMA_LOCAL_CUDA_CUH
