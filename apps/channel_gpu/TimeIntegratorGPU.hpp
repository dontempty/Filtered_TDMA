#ifndef CHANNEL_TIME_INTEGRATOR_GPU_HPP
#define CHANNEL_TIME_INTEGRATOR_GPU_HPP

#include "DeviceField.hpp"
#include "RestartIO.hpp"

namespace channel {

class BoundaryConditionGPU;
class ChannelForcingGPU;
class Config;
class FieldOutput;
class Grid;
class HaloExchangerGPU;
class MomentumSolverGPU;
class MpiTopology;
class PressureSolverGPU;
class StatisticsGPU;
class Subdomain;

class TimeIntegratorGPU {
public:
    TimeIntegratorGPU(const Config& cfg, const MpiTopology& topo, const Subdomain& sub,
                      const Grid& grid, const HaloExchangerGPU& halo,
                      const BoundaryConditionGPU& bc, MomentumSolverGPU& momentum,
                      PressureSolverGPU& pressure, ChannelForcingGPU& forcing,
                      StatisticsGPU& stats, RestartIO& restart, FieldOutput& field_out);

    void run(DeviceField& U, DeviceField& V, DeviceField& W, DeviceField& P,
             RestartState& state);

private:
    double max_div_host_(const DeviceField& U, const DeviceField& V, const DeviceField& W) const;
    double wss_host_(const DeviceField& U) const;
    void copy_fields_to_host_(const DeviceField& U, const DeviceField& V,
                              const DeviceField& W, const DeviceField& P,
                              Field<double>& hU, Field<double>& hV,
                              Field<double>& hW, Field<double>& hP) const;

    const Config& cfg_;
    const MpiTopology& topo_;
    const Subdomain& sub_;
    const Grid& grid_;
    const HaloExchangerGPU& halo_;
    const BoundaryConditionGPU& bc_;
    MomentumSolverGPU& momentum_;
    PressureSolverGPU& pressure_;
    ChannelForcingGPU& forcing_;
    StatisticsGPU& stats_;
    RestartIO& restart_;
    FieldOutput& field_out_;
};

} // namespace channel

#endif // CHANNEL_TIME_INTEGRATOR_GPU_HPP
