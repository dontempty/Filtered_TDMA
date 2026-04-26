// channel/MomentumSolver.hpp
//
// Beam-Warming + 3-stage ADI velocity update (PaScaL_TCS style).
// Both diffusion and linearized convection are implicit in the TDMA matrix.
// Each ADI sweep uses FilteredTdmaSolver across the relevant sub-communicator,
// supporting full 3D domain decomposition (np1 × np2 × np3).

#ifndef CHANNEL_MOMENTUM_SOLVER_HPP
#define CHANNEL_MOMENTUM_SOLVER_HPP

#include <memory>
#include <vector>

#include "Field.hpp"
#include "TdmaSolver.hpp"

namespace channel {

class MpiTopology;
class Subdomain;
class Grid;
class HaloExchanger;
struct Config;

class MomentumSolver {
public:
    MomentumSolver(const Config& cfg,
                   const MpiTopology& topo,
                   const Subdomain& sub,
                   const Grid& grid,
                   const HaloExchanger& halo);

    void advance(Field<double>& U, Field<double>& V, Field<double>& W,
                 const Field<double>& P,
                 double dt, double mean_dPdx);

private:
    enum Component { COMP_U, COMP_V, COMP_W };

    void compute_rhs_(Component which,
                      Field<double>& dQ,
                      const Field<double>& U,
                      const Field<double>& V,
                      const Field<double>& W,
                      const Field<double>& P,
                      double dt, double mean_dPdx);

    void adi_sweep_x_(Component which, Field<double>& dQ, double dt,
                      const Field<double>& U, const Field<double>& V, const Field<double>& W);
    void adi_sweep_y_(Component which, Field<double>& dQ, double dt,
                      const Field<double>& U, const Field<double>& V, const Field<double>& W);
    void adi_sweep_z_(Component which, Field<double>& dQ, double dt,
                      const Field<double>& U, const Field<double>& V, const Field<double>& W);

    // Beam-Warming cross-component coupling — UPPER triangle (post-ADI).
    // Mirrors MPM-STD blockLdU / blockLdV (core_momentum.f90:535-566).
    //   V eq:   dV -= dt · M23 · dW          (W → V in z)
    //   U eq:   dU -= dt · (M12·dV + M13·dW) (V → U in y, W → U in z)
    void cross_BW_V_(Field<double>& dV, const Field<double>& V,
                     const Field<double>& dW, double dt);
    void cross_BW_U_(Field<double>& dU, const Field<double>& U,
                     const Field<double>& dV, const Field<double>& dW, double dt);

    // Beam-Warming cross-component coupling — LOWER triangle (pre-ADI Gauss-Seidel).
    // Mirrors the "M21ddU" / "M31ddU" / "M32ddV" sections embedded inside
    // MPM-STD's solvedV/solvedW Amatrix kernels (core_momentum.f90:1222-1229
    // and 1536-1552).  Subtracts dt·M_ji·dQ_j from the RHS of equation i,
    // using freshly-computed increments dQ_j (j < i in the GS chain U→V→W).
    //   V eq:  dV_rhs -= dt · M21 · dU                        (U → V in x)
    //   W eq:  dW_rhs -= dt · M31 · dU                        (U → W in x)
    //   W eq:  dW_rhs -= dt · M32 · dV                        (V → W in y)
    void cross_BW_V_M21_(Field<double>& dV_rhs, const Field<double>& V,
                         const Field<double>& dU, double dt);
    void cross_BW_W_M31_(Field<double>& dW_rhs, const Field<double>& W,
                         const Field<double>& dU, double dt);
    void cross_BW_W_M32_(Field<double>& dW_rhs, const Field<double>& W,
                         const Field<double>& dV, double dt);

    const Config*        cfg_  = nullptr;
    const Subdomain*     sub_  = nullptr;
    const Grid*          grid_ = nullptr;
    const HaloExchanger* halo_ = nullptr;

    double inv_Re_ = 0.0;

    int np3_ = 1;   // needed to check z-BC boundary rank
    int rank_z_ = 0;

    // Scratch velocity increment fields
    Field<double> dU_, dV_, dW_;

    // Tridiagonal solver per axis (shared across U/V/W components).
    // Backend (FILTERED or PASCAL) selected via cfg.tdma_backend.
    std::unique_ptr<TdmaSolver> fdma_x_, fdma_y_, fdma_z_;

    // Pre-allocated coefficient arrays (avoid per-step allocation)
    // x-sweep: [nx_loc × (ny_loc*nz_loc)]
    std::vector<double> Ax_, Bx_, Cx_, Dx_;
    // y-sweep: [ny_loc × (nx_loc*nz_loc)]
    std::vector<double> Ay_, By_, Cy_, Dy_;
    // z-sweep: [nz_loc × (nx_loc*ny_loc)]
    std::vector<double> Az_, Bz_, Cz_, Dz_;

    int nx_ = 0, ny_ = 0, nz_ = 0;

    long   step_count_ = 0;     // incremented each advance() call
    double tdma_time_     = 0.0;  // accumulated wall-time of TDMA solves   (step > 20000)
    double momentum_time_ = 0.0;  // accumulated wall-time of full advance() (step > 20000)

public:
    double tdma_time()     const { return tdma_time_; }
    double momentum_time() const { return momentum_time_; }
};

} // namespace channel

#endif // CHANNEL_MOMENTUM_SOLVER_HPP
