// channel/MomentumSolver.hpp
//
// Projection-method momentum solver — v2 style:
//   1. Explicit conservative advection N(u,v,w)  (divergence form on staggered grid)
//   2. AB2 update:  u_tilde = u + dt*(1.5*N^n - 0.5*N^{n-1})    (Euler on first step)
//   3. Body force:  u_tilde += -dt*mean_dPdx                     (U component only)
//   4. CN diffusion ADI:
//        rhs = u_tilde + (nu*dt/2) * lap(u_tilde)
//        (I - nu*dt/2*∂²/∂z²)(I - nu*dt/2*∂²/∂y²)(I - nu*dt/2*∂²/∂x²) u* = rhs
//      Wall BC in z: U/V antisymmetric ghost (fold into diagonal),
//                    W zero ghost (off-diagonal zeroed at wall row).
//   5. (Pressure projection done by PressureSolver afterwards.)
//
// Reference: /shared/home/wel1come1234/workspace/TDMA/Filtered_TDMAv2/channel/
//            (advection.cpp / diffusion.cpp / projection.cpp)
//
// Original Beam-Warming version preserved as MomentumSolver_BW.{hpp,cpp}.bak.

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

    // P is unused here (Chorin-style: pressure handled fully in PressureSolver).
    // Signature kept for compatibility with the existing TimeIntegrator call.
    void advance(Field<double>& U, Field<double>& V, Field<double>& W,
                 const Field<double>& P,
                 double dt, double mean_dPdx);

private:
    enum Component { COMP_U, COMP_V, COMP_W };

    // Conservative-form advection on staggered grid (v2 advection.cpp port).
    // Convention here: U at x-face i-1/2, V at y-face j-1/2, W at z-face k-1/2.
    void compute_advection_(const Field<double>& U,
                            const Field<double>& V,
                            const Field<double>& W,
                            Field<double>& Nu,
                            Field<double>& Nv,
                            Field<double>& Nw) const;

    // u_tilde = u + dt*(1.5*N_new - 0.5*N_old)   (or u + dt*N_new on first step)
    void apply_AB2_(Field<double>& U, Field<double>& V, Field<double>& W,
                    const Field<double>& Nu_new, const Field<double>& Nv_new,
                    const Field<double>& Nw_new,
                    const Field<double>& Nu_old, const Field<double>& Nv_old,
                    const Field<double>& Nw_old,
                    double dt, bool first) const;

    // rhs = q + (nu*dt/2) * lap(q)   (component-aware: U/V/W staggering only
    // affects which dz / dmz is used in the z-direction).
    void add_explicit_lap_(Component which, const Field<double>& q,
                           Field<double>& rhs, double nu_dt_half) const;

    // Solve (I - nu*dt/2 * ∂²/∂x²) q_out = q_in via FilteredTdmaSolver.
    // adi_x_ optionally unpacks the final TDMA result into `dst` instead of q,
    // letting the caller skip a full-field copy when q is a scratch buffer.
    void adi_x_(Component which, Field<double>& q, double nu_dt_half,
                Field<double>* dst = nullptr);
    void adi_y_(Component which, Field<double>& q, double nu_dt_half);
    void adi_z_(Component which, Field<double>& q, double nu_dt_half);

    const Config*        cfg_  = nullptr;
    const Subdomain*     sub_  = nullptr;
    const Grid*          grid_ = nullptr;
    const HaloExchanger* halo_ = nullptr;

    double inv_Re_ = 0.0;

    int np3_ = 1;
    int rank_z_ = 0;

    // AB2 history (advection N at previous and current step).
    Field<double> Nu_old_, Nv_old_, Nw_old_;
    Field<double> Nu_new_, Nv_new_, Nw_new_;
    bool first_step_ = true;

    // Scratch RHS buffer (re-used across U/V/W).
    Field<double> rhs_;

    // Tridiagonal solver per axis.
    std::unique_ptr<TdmaSolver> fdma_x_, fdma_y_, fdma_z_;

    // Pre-allocated TDMA coefficient/RHS arrays.
    std::vector<double> Ax_, Bx_, Cx_, Dx_;
    std::vector<double> Ay_, By_, Cy_, Dy_;
    std::vector<double> Az_, Bz_, Cz_, Dz_;

    int nx_ = 0, ny_ = 0, nz_ = 0;

    long   step_count_    = 0;
    double tdma_time_     = 0.0;
    double momentum_time_ = 0.0;

public:
    double tdma_time()     const { return tdma_time_; }
    double momentum_time() const { return momentum_time_; }
};

} // namespace channel

#endif // CHANNEL_MOMENTUM_SOLVER_HPP
