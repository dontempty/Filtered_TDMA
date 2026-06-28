#ifndef PARA_RANGE_HPP
#define PARA_RANGE_HPP

/// Compute the local index range [ista, iend] for a given MPI rank.
/// Splits the global range [start, end] evenly across nproc processes.
void para_range(int start, int end, int nproc, int rank, int& ista, int& iend);

#endif // PARA_RANGE_HPP
