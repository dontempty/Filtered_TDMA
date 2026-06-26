#ifndef CHANNEL_HALO_EXCHANGER_GPU_HPP
#define CHANNEL_HALO_EXCHANGER_GPU_HPP

#include <mpi.h>

#include "DeviceBuffer.hpp"
#include "DeviceField.hpp"

namespace channel {

class MpiTopology;
class Subdomain;

class HaloExchangerGPU {
public:
    HaloExchangerGPU(const MpiTopology& topo, const Subdomain& sub);
    void exchange(DeviceField& f) const;
    void exchange_axis(DeviceField& f, int axis) const;

private:
    const MpiTopology& topo_;
    int nx_ = 0, ny_ = 0, nz_ = 0;
    mutable DeviceBuffer<double> send_lo_[3], send_hi_[3], recv_lo_[3], recv_hi_[3];
};

} // namespace channel

#endif // CHANNEL_HALO_EXCHANGER_GPU_HPP
