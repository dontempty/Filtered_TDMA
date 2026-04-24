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

    // MASS_FLOW: enforce bulk velocity by a direct constant shift.
    //
    // Adding a spatially-uniform constant to U does not change div(U),
    // so this can safely be applied at any point in the time loop.
    //
    // The equivalent pressure gradient increment is -dU/dt, but when
    // dt is very small (CFL-limited) that ratio explodes. We instead
    // clamp the dPdx update independently of the velocity shift.

    const double Ub  = bulk_velocity_(U);
    const double dU  = cfg_.target_bulk_velocity - Ub;   // velocity correction

    // Apply direct shift to every cell (preserves divergence-free field)
    const int nx = sub_.nx(), ny = sub_.ny(), nz = sub_.nz();
    for (int k = 1; k <= nz; ++k)
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i)
                U(i, j, k) += dU;

    // Update dPdx_ with a bounded increment.
    // For Poiseuille flow at Re_b: dPdx_lam = -12 * nu / h^2
    // where h = Lz/2 (half-channel height).  We use this as scale.
    const double half_h  = 0.5 * cfg_.Lz;
    const double nu      = 1.0 / cfg_.Re_b;
    const double dPdx_lam = 12.0 * nu / (half_h * half_h);   // O(1) scale

    // The instantaneous equivalent increment: -dU/dt
    // Clamp to ±50 × laminar value so dt doesn't amplify noise
    double raw_inc = (dt > 1.0e-15) ? (-dU / dt) : 0.0;
    double max_inc = 50.0 * dPdx_lam;
    double clamped_inc = std::max(-max_inc, std::min(max_inc, raw_inc));

    dPdx_ += clamped_inc;

    // Also clamp the accumulated dPdx_ itself to prevent run-away
    double max_dPdx = 200.0 * dPdx_lam;
    dPdx_ = std::max(-max_dPdx, std::min(max_dPdx, dPdx_));

    return dPdx_;
}

} // namespace channel
