#ifndef CHANNEL_MOMENTUM_SOLVER_GPU_HPP
#define CHANNEL_MOMENTUM_SOLVER_GPU_HPP

#include <memory>
#include <string>
#include <vector>

#include "DeviceBuffer.hpp"
#include "DeviceField.hpp"
#include "TdmaSolverGPU.hpp"

namespace channel {

class Config;
class Grid;
class HaloExchangerGPU;
class MpiTopology;
class Subdomain;

class MomentumSolverGPU {
public:
    MomentumSolverGPU(const Config& cfg, const MpiTopology& topo,
                      const Subdomain& sub, const Grid& grid,
                      const HaloExchangerGPU& halo);

    void advance(DeviceField& U, DeviceField& V, DeviceField& W,
                 const DeviceField& P, double dt, const double* d_mean_dPdx);

    double tdma_x_time() const { return tdma_x_time_; }
    double tdma_y_time() const { return tdma_y_time_; }
    double tdma_z_time() const { return tdma_z_time_; }
    double tdma_time() const { return tdma_x_time_ + tdma_y_time_ + tdma_z_time_; }
    double momentum_time() const { return momentum_time_; }
    void write_timing_csv(const std::string& path) const;

    /// Truncation depth J used by the most recent x/y/z TDMA solve
    /// (FILTERED backend only; -1 for PASCAL).
    int last_Jx() const { return fdma_x_->last_J(); }
    int last_Jy() const { return fdma_y_->last_J(); }
    int last_Jz() const { return fdma_z_->last_J(); }

private:
    enum Component { COMP_U = 0, COMP_V = 1, COMP_W = 2 };

    void adi_z_(Component c, DeviceField& q, double nu_dt_half);
    void adi_y_(Component c, DeviceField& q, double nu_dt_half);
    void adi_x_(Component c, DeviceField& q, double nu_dt_half, DeviceField& dst);

    const Config& cfg_;
    const HaloExchangerGPU& halo_;
    int nx_ = 0, ny_ = 0, nz_ = 0;
    int np3_ = 1, rank_z_ = 0;
    double inv_Re_ = 0.0;
    long step_count_ = 0;
    bool first_step_ = true;

    DeviceField Nu_old_, Nv_old_, Nw_old_;
    DeviceField Nu_new_, Nv_new_, Nw_new_;
    DeviceField rhs_;

    DeviceBuffer<double> dx_[3], dmx_[3];
    DeviceBuffer<double> Ax_, Bx_, Cx_, Dx_;
    DeviceBuffer<double> Ay_, By_, Cy_, Dy_;
    DeviceBuffer<double> Az_, Bz_, Cz_, Dz_;

    std::unique_ptr<TdmaSolverGPU> fdma_x_, fdma_y_, fdma_z_;

    double tdma_x_time_ = 0.0, tdma_y_time_ = 0.0, tdma_z_time_ = 0.0;
    double momentum_time_ = 0.0;
    std::vector<long> timing_step_;
    std::vector<double> timing_x_, timing_y_, timing_z_, timing_mom_;
};

} // namespace channel

#endif // CHANNEL_MOMENTUM_SOLVER_GPU_HPP
