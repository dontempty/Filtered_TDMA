#ifndef TDMA_LOCAL_HPP
#define TDMA_LOCAL_HPP

#include <cstddef>
#include <vector>

/// Local (single-rank) Thomas algorithm for multiple RHS systems.
/// Layout: row-major [n_row × n_sys], i.e. row j starts at offset j*n_sys.
void tdma_many(double* __restrict A,
               double* __restrict B,
               double* __restrict C,
               double* __restrict D,
               int n_sys, int n_row);

/// Local Thomas algorithm for a single tridiagonal system of size n.
/// Vectors a, b, c, d have size n; d is overwritten with the solution.
void tdma_single(std::vector<double>& a, std::vector<double>& b,
                 std::vector<double>& c, std::vector<double>& d, int n);

/// Local cyclic Thomas algorithm for a single tridiagonal system of size n.
/// Handles periodic (cyclic) boundary conditions.
void tdma_cyclic_single(std::vector<double>& a, std::vector<double>& b,
                        std::vector<double>& c, std::vector<double>& d, int n);

/// Local cyclic Thomas algorithm for multiple RHS systems.
/// Extends tdma_cyclic_single to n_sys simultaneous systems.
/// Layout: row-major [n_row × n_sys], i.e. row j starts at offset j*n_sys.
void tdma_cyclic_many(double* __restrict A,
                      double* __restrict B,
                      double* __restrict C,
                      double* __restrict D,
                      int n_sys, int n_row);

#endif // TDMA_LOCAL_HPP
