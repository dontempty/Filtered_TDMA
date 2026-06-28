#ifndef TDMA_LOCAL_HPP
#define TDMA_LOCAL_HPP

#include <cstddef>

/// Local (single-rank) Thomas algorithm for multiple RHS systems.
/// Layout: row-major [n_row × n_sys], i.e. row j starts at offset j*n_sys.

void ftdma_many(double* __restrict A,
               double* __restrict B,
               double* __restrict C,
               double* __restrict D,
               int n_sys, int n_row);

void ftdma_cyclic_many(double* __restrict A,
                    double* __restrict B,
                    double* __restrict C,
                    double* __restrict D,
                    int n_sys, int n_row);

#endif // TDMA_LOCAL_HPP
