#ifndef CHANNEL_FORCING_GPU_HPP
#define CHANNEL_FORCING_GPU_HPP

#include "DeviceBuffer.hpp"
#include "DeviceField.hpp"

namespace channel {

class Config;
class Grid;
class MpiTopology;
class Subdomain;

class ChannelForcingGPU {
public:
    ChannelForcingGPU(const Config& cfg, const MpiTopology& topo,
                      const Subdomain& sub, const Grid& grid);

    const double* device_mean_dPdx() const { return d_dPdx_.data(); }
    double mean_dPdx_host() const;
    void set_mean_dPdx(double v);
    void correct(DeviceField& U, double dt);
    double bulk_velocity_host(const DeviceField& U) const;

private:
    const Config& cfg_;
    const MpiTopology& topo_;
    int nx_ = 0, ny_ = 0, nz_ = 0;
    double total_volume_ = 1.0;
    DeviceBuffer<double> dx_, dy_, dz_;
    mutable DeviceBuffer<double> d_part_, d_sum_, d_global_sum_;
    DeviceBuffer<double> d_dPdx_;
};

} // namespace channel

#endif // CHANNEL_FORCING_GPU_HPP
