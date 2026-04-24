// channel/HaloExchanger.hpp
//
// One-deep halo (ghost cell) exchange in all six directions, using
// MPI_Type_create_subarray derived datatypes for zero-copy face slabs.
//
// Periodicity is taken from the Cartesian topology: non-periodic
// neighbors collapse to MPI_PROC_NULL and the corresponding Isend/Irecv
// becomes a no-op (so wall faces are left untouched here; explicit BC
// values are applied separately by BoundaryCondition).
//
// PaScaL_TCS analogue: module_mpi_subdomain.f90 (DDT setup + ghostcell_update)

#ifndef CHANNEL_HALO_EXCHANGER_HPP
#define CHANNEL_HALO_EXCHANGER_HPP

#include <mpi.h>

#include "Field.hpp"

namespace channel {

class MpiTopology;
class Subdomain;

class HaloExchanger {
public:
    HaloExchanger(const MpiTopology& topo, const Subdomain& sub);
    ~HaloExchanger();

    HaloExchanger(const HaloExchanger&)            = delete;
    HaloExchanger& operator=(const HaloExchanger&) = delete;

    /// Exchange the one-cell halo along all three axes for `f`.
    /// `f` must have interior dims matching the Subdomain used at construction.
    void exchange(Field<double>& f) const;

    /// Exchange along a single axis (0=x, 1=y, 2=z).
    void exchange_axis(Field<double>& f, int axis) const;

private:
    void build_subarray_(int axis, int nx, int ny, int nz);

    const MpiTopology& topo_;

    // Per-axis face datatypes:
    //   send_lo_[a] = subarray on the low-side  interior plane (i=1)
    //   send_hi_[a] = subarray on the high-side interior plane (i=n)
    //   recv_lo_[a] = subarray on the low-side  ghost  plane (i=0)
    //   recv_hi_[a] = subarray on the high-side ghost  plane (i=n+1)
    MPI_Datatype send_lo_[3] = {MPI_DATATYPE_NULL, MPI_DATATYPE_NULL, MPI_DATATYPE_NULL};
    MPI_Datatype send_hi_[3] = {MPI_DATATYPE_NULL, MPI_DATATYPE_NULL, MPI_DATATYPE_NULL};
    MPI_Datatype recv_lo_[3] = {MPI_DATATYPE_NULL, MPI_DATATYPE_NULL, MPI_DATATYPE_NULL};
    MPI_Datatype recv_hi_[3] = {MPI_DATATYPE_NULL, MPI_DATATYPE_NULL, MPI_DATATYPE_NULL};

    int nx_ = 0, ny_ = 0, nz_ = 0;   // interior counts
};

} // namespace channel

#endif // CHANNEL_HALO_EXCHANGER_HPP
