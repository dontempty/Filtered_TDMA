// channel/TimeIntegrator.hpp
//
// Top-level time loop: orchestrates momentum, pressure, forcing, statistics,
// field output, and restart IO.

#ifndef CHANNEL_TIME_INTEGRATOR_HPP
#define CHANNEL_TIME_INTEGRATOR_HPP

#include "Field.hpp"
#include <utility>

namespace channel {

class MpiTopology;
class Subdomain;
class Grid;
class HaloExchanger;
class BoundaryCondition;
class MomentumSolver;
class PressureSolver;
class ChannelForcing;
class Statistics;
class RestartIO;
class FieldOutput;
struct Config;
struct RestartState;

class TimeIntegrator {
public:
    TimeIntegrator(const Config& cfg,
                   const MpiTopology& topo,
                   const Subdomain& sub,
                   const Grid& grid,
                   const HaloExchanger& halo,
                   const BoundaryCondition& bc,
                   MomentumSolver& momentum,
                   PressureSolver& pressure,
                   ChannelForcing& forcing,
                   Statistics& stats,
                   RestartIO& restart,
                   FieldOutput& field_out);

    void run(Field<double>& U, Field<double>& V, Field<double>& W,
             Field<double>& P, RestartState& state);

private:
    double cfl_dt_(const Field<double>& U, const Field<double>& V,
                   const Field<double>& W) const;

    // Returns {rho_max (dz_min, wall), rho_min (dz_max, centre)}
    std::pair<double,double> rho_diagnostic_(double dt) const;

    double wss_diagnostic_(const Field<double>& U) const;

    double bulk_velocity_(const Field<double>& U) const;

    double max_div_u_(const Field<double>& U, const Field<double>& V,
                      const Field<double>& W) const;

    // Debug: locate the cell with global max |U|, |V|, |W| (and global divU peak)
    // and append one row per call to a per-rank CSV under dir_statistics/.
    // If `verbose_stdout` is true, rank 0 also prints a formatted block to
    // stdout (used by the divU-overflow abort path).  Pass fp = nullptr to
    // skip the CSV write while still emitting the stdout dump.
    void write_max_velocity_debug_(const Field<double>& U,
                                   const Field<double>& V,
                                   const Field<double>& W,
                                   long step, double dt, double t,
                                   std::FILE* fp,
                                   bool verbose_stdout = false) const;

    const Config&           cfg_;
    const MpiTopology&      topo_;
    const Subdomain&        sub_;
    const Grid&             grid_;
    const HaloExchanger&    halo_;
    const BoundaryCondition& bc_;
    MomentumSolver&         momentum_;
    PressureSolver&         pressure_;
    ChannelForcing&         forcing_;
    Statistics&             stats_;
    RestartIO&              restart_;
    FieldOutput&            field_out_;
};

} // namespace channel

#endif // CHANNEL_TIME_INTEGRATOR_HPP
