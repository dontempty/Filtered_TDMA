#ifndef CHANNEL_BOUNDARY_CONDITION_GPU_HPP
#define CHANNEL_BOUNDARY_CONDITION_GPU_HPP

#include "DeviceField.hpp"

namespace channel {

class MpiTopology;
class Subdomain;

class BoundaryConditionGPU {
public:
    BoundaryConditionGPU(const MpiTopology& topo, const Subdomain& sub);
    void apply(DeviceField& U, DeviceField& V, DeviceField& W) const;

private:
    int nx_ = 0, ny_ = 0, nz_ = 0;
    bool low_ = false;
    bool high_ = false;
};

} // namespace channel

#endif // CHANNEL_BOUNDARY_CONDITION_GPU_HPP
