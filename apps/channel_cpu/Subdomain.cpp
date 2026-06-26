#include "Subdomain.hpp"

#include "Config.hpp"
#include "MpiTopology.hpp"
#include "para_range.hpp"

#include <cstdio>
#include <mpi.h>

namespace channel {

Subdomain::Subdomain(const Config& cfg, const MpiTopology& topo)
{
    global_n_[0] = cfg.n1m;
    global_n_[1] = cfg.n2m;
    global_n_[2] = cfg.n3m;

    for (int a = 0; a < 3; ++a) {
        int ista = 0, iend = -1;
        para_range(1, global_n_[a], topo.size_in(a), topo.rank_in(a), ista, iend);
        ista_[a] = ista;
        iend_[a] = iend;
        n_[a]    = iend - ista + 1;
    }
}

void Subdomain::print(const MpiTopology& topo) const
{
    int rank = topo.rank();
    int nprocs = topo.nprocs();
    for (int r = 0; r < nprocs; ++r) {
        if (r == rank) {
            std::printf("[rank %3d] coord=(%d,%d,%d) "
                        "x=[%d,%d](nx=%d) y=[%d,%d](ny=%d) z=[%d,%d](nz=%d)\n",
                        rank, topo.coord(0), topo.coord(1), topo.coord(2),
                        ista_[0], iend_[0], n_[0],
                        ista_[1], iend_[1], n_[1],
                        ista_[2], iend_[2], n_[2]);
            std::fflush(stdout);
        }
        MPI_Barrier(topo.cart());
    }
}

} // namespace channel
