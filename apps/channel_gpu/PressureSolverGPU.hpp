#ifndef CHANNEL_PRESSURE_SOLVER_GPU_HPP
#define CHANNEL_PRESSURE_SOLVER_GPU_HPP

#include <cufft.h>
#include <memory>
#include <vector>

#include "DeviceBuffer.hpp"
#include "DeviceField.hpp"
#include "pascal_tdma_cuda.hpp"

namespace channel {

class Config;
class Grid;
class HaloExchangerGPU;
class MpiTopology;
class Subdomain;

class PressureSolverGPU {
public:
    PressureSolverGPU(const Config& cfg, const MpiTopology& topo,
                      const Subdomain& sub, const Grid& grid,
                      const HaloExchangerGPU& halo);
    ~PressureSolverGPU();

    void solve(DeviceField& U, DeviceField& V, DeviceField& W,
               DeviceField& P, double dt);

private:
    void compute_rhs_(const DeviceField& U, const DeviceField& V,
                      const DeviceField& W, double dt);
    void transpose_C_to_I_();
    void transpose_I_to_C_();
    void transpose_I_to_Y_();
    void transpose_Y_to_I_();
    void solve_tdma_z_();
    void debug_poisson_residual_() const;
    void project_(DeviceField& U, DeviceField& V, DeviceField& W,
                  DeviceField& P, double dt);

    const MpiTopology& topo_;
    const HaloExchangerGPU& halo_;
    int np1_ = 1, np2_ = 1, np3_ = 1;
    int nx_loc_ = 0, ny_loc_ = 0, nz_loc_ = 0;
    int n1m_ = 0, n2m_ = 0, n3m_ = 0, Nxh_ = 0, n3_I_ = 0;
    int h1p_Y_me_ = 0, ix_start_Y_me_ = 0, n_sys_Y_ = 0;
    int rank_xz_ = 0, size_xz_ = 1;

    std::vector<int> h1p_Y_, ix_start_Y_;
    std::vector<int> scnts_Y_, sdsp_Y_, rcnts_Y_, rdsp_Y_;
    std::vector<int> scnts_Yb_, sdsp_Yb_, rcnts_Yb_, rdsp_Yb_;

    DeviceBuffer<double> dx_, dy_, dz_, dmx_, dmy_, dmz_;
    DeviceBuffer<double> dz_g_, dmz_g_, kx2_, ky2_;
    DeviceBuffer<double> rhs_C_, rhs_I_, dp_I_;
    DeviceBuffer<cufftDoubleComplex> hat_I_, hat_Y_;
    DeviceField dp_;
    DeviceBuffer<double> tdma_Am_, tdma_Ac_, tdma_Ap_;
    DeviceBuffer<double> tdma_Am2_, tdma_Ac2_, tdma_Ap2_;
    DeviceBuffer<double> tdma_Be_r_, tdma_Be_c_;
    DeviceBuffer<double> tx_sbuf_C_, tx_rbuf_C_, tx_sbuf_Y_, tx_rbuf_Y_;

    cufftHandle plan_fwd_x_ = 0, plan_bwd_x_ = 0, plan_y_ = 0;
    std::unique_ptr<PaScaLTDMACUDA> tdma_z_;
};

} // namespace channel

#endif // CHANNEL_PRESSURE_SOLVER_GPU_HPP
