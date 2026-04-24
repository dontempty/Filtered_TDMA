// channel/Subdomain.hpp
//
// Per-rank subdomain extents, derived by splitting a global cell-centered
// grid (n1m × n2m × n3m) over the (np1 × np2 × np3) Cartesian topology.
//
// PaScaL_TCS analogue: module_mpi_subdomain.f90 (the index sets
// n*msub, ista/iend; DDT construction is left to HaloExchanger).

#ifndef CHANNEL_SUBDOMAIN_HPP
#define CHANNEL_SUBDOMAIN_HPP

namespace channel {

class MpiTopology;
struct Config;

class Subdomain {
public:
    Subdomain(const Config& cfg, const MpiTopology& topo);

    // Interior-cell counts on this rank (excluding halos).
    int nx() const { return n_[0]; }
    int ny() const { return n_[1]; }
    int nz() const { return n_[2]; }

    // Global cell index range (1-based, inclusive) of interior cells.
    int ista(int axis) const { return ista_[axis]; }
    int iend(int axis) const { return iend_[axis]; }

    // Global cell counts (n1m, n2m, n3m).
    int global_n(int axis) const { return global_n_[axis]; }

    void print(const MpiTopology& topo) const;

private:
    int n_[3]        = {0, 0, 0};
    int ista_[3]     = {1, 1, 1};
    int iend_[3]     = {0, 0, 0};
    int global_n_[3] = {0, 0, 0};
};

} // namespace channel

#endif // CHANNEL_SUBDOMAIN_HPP
