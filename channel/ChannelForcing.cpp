#include "ChannelForcing.hpp"

#include "Grid.hpp"
#include "MpiTopology.hpp"
#include "Subdomain.hpp"

#include <cmath>
#include <mpi.h>

namespace channel {

ChannelForcing::ChannelForcing(const Config& cfg,
                               const MpiTopology& topo,
                               const Subdomain& sub,
                               const Grid& grid)
    : cfg_(cfg), topo_(topo), sub_(sub), grid_(grid)
{
    if (cfg_.forcing_mode == ForcingMode::PRESSURE_GRADIENT)
        dPdx_ = cfg_.target_dPdx;

    // total domain volume — fixed
    total_volume_ = cfg_.Lx * cfg_.Ly * cfg_.Lz;
}

double ChannelForcing::bulk_velocity_(const Field<double>& U) const
{
    const int nx = sub_.nx(), ny = sub_.ny(), nz = sub_.nz();
    const auto& dx = grid_.dx(0);
    const auto& dy = grid_.dx(1);
    const auto& dz = grid_.dx(2);

    double local = 0.0;
    for (int k = 1; k <= nz; ++k)
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i)
                local += U(i, j, k) * dx[i] * dy[j] * dz[k];

    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, topo_.cart());
    return global / total_volume_;
}

double ChannelForcing::correct(Field<double>& U, double dt)
{
    if (cfg_.forcing_mode == ForcingMode::PRESSURE_GRADIENT)
        return dPdx_;       // constant — applied via momentum RHS

    // MASS_FLOW: MPM-STD cuda_momentum_masscorrection pattern.
    //   Ub_pseudo = bulk(u*).
    //   DMpresg   = (Ub_pseudo − Ub_target) / dt   (with sign convention
    //               matching MPM-STD: presgrad1 += DMpresg, then
    //               body force term in next-step RHS is −presgrad1).
    //   Apply uniform shift  u ← u − dt·DMpresg = u + (Ub_target − Ub_pseudo)
    //   so that bulk(u) = Ub_target after the shift.
    // No clamping (matches MPM-STD): the increment is the exact instantaneous
    // pressure-gradient correction needed; clamping would introduce phase lag.
    const double Ub        = bulk_velocity_(U);
    const double DMpresg   = (dt > 1.0e-15) ? (Ub - cfg_.target_bulk_velocity) / dt : 0.0;
    const double shift     = -dt * DMpresg;       // = Ub_target − Ub

    const int nx = sub_.nx(), ny = sub_.ny(), nz = sub_.nz();
    for (int k = 1; k <= nz; ++k)
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i)
                U(i, j, k) += shift;

    dPdx_ += DMpresg;
    return dPdx_;
}

} // namespace channel
