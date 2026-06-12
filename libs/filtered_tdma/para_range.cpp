#include "para_range.hpp"
#include <algorithm>

void para_range(int start, int end, int nproc, int rank, int &ista, int &iend) {
    int len  = end - start + 1;
    int base = len / nproc;
    int rem  = len % nproc;
    ista = start + rank * base + std::min(rank, rem);
    iend = ista + base - 1 + (rank < rem ? 1 : 0);
}
