#ifndef CHANNEL_STATISTICS_GPU_HPP
#define CHANNEL_STATISTICS_GPU_HPP

#include <string>
#include <vector>

#include "DeviceBuffer.hpp"
#include "DeviceField.hpp"

namespace channel {

class Grid;
class MpiTopology;
class Subdomain;

class StatisticsGPU {
public:
    StatisticsGPU(const MpiTopology& topo, const Subdomain& sub, const Grid& grid);

    void accumulate(const DeviceField& U, const DeviceField& V,
                    const DeviceField& W, const DeviceField& P);
    void write(const std::string& path, int step, double Re_b, bool reset_after = false);
    void reset();
    long samples() const { return n_; }

private:
    const MpiTopology& topo_;
    int nx_ = 0, ny_ = 0, nz_local_ = 0, nz_global_ = 0, kstart_ = 0;
    int nx_global_ = 0, ny_global_ = 0;
    long n_ = 0;
    DeviceBuffer<double> dx_, dy_, dz_;
    DeviceBuffer<double> U_m_, U2_m_, V_m_, V2_m_, Wc_m_, Wc2_m_, P_m_;
    DeviceBuffer<double> Ug_m_, Wg_m_, UWg_m_;
    std::vector<double> zc_global_, z_face_global_;
};

} // namespace channel

#endif // CHANNEL_STATISTICS_GPU_HPP
