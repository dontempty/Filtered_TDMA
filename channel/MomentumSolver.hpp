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
struct Config;

class MomentumSolver {
public:
    MomentumSolver(const Config& cfg,
                   const MpiTopology& topo,
                   const Subdomain& sub,
                   const Grid& grid);

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

    // BW cross-component RHS injection (N'^n off-diagonal). Called after
    // compute_rhs_ and before ADI sweeps. Uses increments from components
    // already solved in this step:
    //   V : uses dU    → adds -0.5*dt*dU*(∂V/∂x)
    //   W : uses dU,dV → adds -0.5*dt*(dU*(∂W/∂x) + dV*(∂W/∂y))
    // U has no prior increment, so it's a no-op for COMP_U.
    void add_cross_BW_(Component which, Field<double>& dQ, double dt,
                       const Field<double>& U, const Field<double>& V, const Field<double>& W,
                       const Field<double>* dU_prev, const Field<double>* dV_prev);

    const Config*    cfg_  = nullptr;
    const Subdomain* sub_  = nullptr;
    const Grid*      grid_ = nullptr;

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
    double tdma_time_  = 0.0;   // accumulated wall-time of TDMA solves (step > 20000)

public:
    double tdma_time() const { return tdma_time_; }
};

} // namespace channel

#endif // CHANNEL_MOMENTUM_SOLVER_HPP
