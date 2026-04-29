// channel/MomentumSolver.cpp  —  v2-style projection momentum (advection + AB2 + 3D Heat ADI)
//
// Port reference: /shared/home/wel1come1234/workspace/TDMA/Filtered_TDMAv2/channel/
//   advection.cpp   — conservative divergence advection on staggered grid
//   diffusion.cpp   — 3D ADI (Z→Y→X) of (I - nu*dt/2*lap) with CN explicit half
//   projection.cpp  — orchestrator (we inline that into advance())
//
// Staggering convention (Filtered_TDMA):
//   U(i,j,k) at x-face i-1/2, y-cell j, z-cell k
//   V(i,j,k) at x-cell i, y-face j-1/2, z-cell k
//   W(i,j,k) at x-cell i, y-cell j, z-face k-1/2
// (v2 uses i+1/2 for u, j+1/2 for v — indices flip on a few stencils accordingly.)
//
// Wall BC (low/high rank, z-direction):
//   U,V : antisymmetric ghost (U(0) = -U(1), U(nz+1) = -U(nz))    → ADI z fold
//   W   : zero at wall face   (W(1) = 0 at low, W(nz+1) = 0 at high)

#include "MomentumSolver.hpp"

#include "Config.hpp"
#include "Grid.hpp"
#include "MpiTopology.hpp"
#include "Subdomain.hpp"
#include "HaloExchanger.hpp"

#include <cmath>
#include <cstring>

namespace channel {

// ---------------------------------------------------------------------------
MomentumSolver::MomentumSolver(const Config& cfg,
                               const MpiTopology& topo,
                               const Subdomain& sub,
                               const Grid& grid,
                               const HaloExchanger& halo)
    : cfg_(&cfg), sub_(&sub), grid_(&grid), halo_(&halo),
      Nu_old_(sub.nx(), sub.ny(), sub.nz()),
      Nv_old_(sub.nx(), sub.ny(), sub.nz()),
      Nw_old_(sub.nx(), sub.ny(), sub.nz()),
      Nu_new_(sub.nx(), sub.ny(), sub.nz()),
      Nv_new_(sub.nx(), sub.ny(), sub.nz()),
      Nw_new_(sub.nx(), sub.ny(), sub.nz()),
      rhs_   (sub.nx(), sub.ny(), sub.nz())
{
    inv_Re_ = 1.0 / cfg.Re_b;
    np3_    = topo.dim(2);
    rank_z_ = topo.rank_in(2);

    nx_ = sub.nx(); ny_ = sub.ny(); nz_ = sub.nz();

    auto backend = TdmaSolver::parse_backend(cfg.tdma_backend);

    // Heat-equation ADI: small constant eps; rho updated each step from coefficients.
    fdma_x_ = std::make_unique<TdmaSolver>(topo, 0, ny_*nz_, nx_, cfg.pbc1, backend, 1.0e-12);
    fdma_y_ = std::make_unique<TdmaSolver>(topo, 1, nx_*nz_, ny_, cfg.pbc2, backend, 1.0e-12);
    fdma_z_ = std::make_unique<TdmaSolver>(topo, 2, nx_*ny_, nz_, /*periodic=*/false, backend, 1.0e-12);

    Ax_.resize((std::size_t)nx_ * ny_ * nz_);
    Bx_.resize((std::size_t)nx_ * ny_ * nz_);
    Cx_.resize((std::size_t)nx_ * ny_ * nz_);
    Dx_.resize((std::size_t)nx_ * ny_ * nz_);

    Ay_.resize((std::size_t)ny_ * nx_ * nz_);
    By_.resize((std::size_t)ny_ * nx_ * nz_);
    Cy_.resize((std::size_t)ny_ * nx_ * nz_);
    Dy_.resize((std::size_t)ny_ * nx_ * nz_);

    Az_.resize((std::size_t)nz_ * nx_ * ny_);
    Bz_.resize((std::size_t)nz_ * nx_ * ny_);
    Cz_.resize((std::size_t)nz_ * nx_ * ny_);
    Dz_.resize((std::size_t)nz_ * nx_ * ny_);
}

// ---------------------------------------------------------------------------
//  Conservative advection — staggered divergence form (v2 advection.cpp port).
//  Index convention here: U at x-face i-1/2, V at y-face j-1/2, W at z-face k-1/2.
//
//  N(u)(i,j,k) = -[ d(uu)/dx + d(vu)/dy + d(wu)/dz ]   (and similarly N(v), N(w))
// ---------------------------------------------------------------------------
void MomentumSolver::compute_advection_(const Field<double>& U,
                                        const Field<double>& V,
                                        const Field<double>& W,
                                        Field<double>& Nu,
                                        Field<double>& Nv,
                                        Field<double>& Nw) const
{
    const int nx = nx_, ny = ny_, nz = nz_;
    const auto& dx  = grid_->dx (0); const auto& dmx = grid_->dmx(0);
    const auto& dy  = grid_->dx (1); const auto& dmy = grid_->dmx(1);
    const auto& dz  = grid_->dx (2); const auto& dmz = grid_->dmx(2);

    // ===== N(u) =================================================
    //  U-CV: x ∈ [xc(i-1), xc(i)] (length dmx[i]),
    //        y full cell j (length dy[j]), z full cell k (length dz[k]).
    for (int k = 1; k <= nz; ++k)
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i) {
                // d(uu)/dx — CV x-faces at cell-centers i-1 and i
                double U_R = 0.5 * (U(i,j,k)   + U(i+1,j,k));   // at xc[i]
                double U_L = 0.5 * (U(i-1,j,k) + U(i,j,k));     // at xc[i-1]
                double flux_uu = (U_R*U_R - U_L*U_L) / dmx[i];

                // d(vu)/dy — CV y-faces at yf[j-1] (bot) and yf[j] (top)
                double V_top = 0.5 * (V(i-1,j+1,k) + V(i,j+1,k));   // V at xf[i-1]
                double V_bot = 0.5 * (V(i-1,j,  k) + V(i,j,  k));
                double u_top = 0.5 * (U(i,j,  k) + U(i,j+1,k));     // U at yf[j]
                double u_bot = 0.5 * (U(i,j-1,k) + U(i,j,  k));
                double flux_vu = (V_top*u_top - V_bot*u_bot) / dy[j];

                // d(wu)/dz — CV z-faces at zf[k-1] (bot) and zf[k] (top)
                double W_top = 0.5 * (W(i-1,j,k+1) + W(i,j,k+1));   // W at xf[i-1]
                double W_bot = 0.5 * (W(i-1,j,k  ) + W(i,j,k  ));
                double u_zt  = 0.5 * (U(i,j,k  ) + U(i,j,k+1));     // U at zf[k]
                double u_zb  = 0.5 * (U(i,j,k-1) + U(i,j,k  ));
                double flux_wu = (W_top*u_zt - W_bot*u_zb) / dz[k];

                Nu(i,j,k) = -(flux_uu + flux_vu + flux_wu);
            }

    // ===== N(v) =================================================
    //  V-CV: x full cell i (length dx[i]),
    //        y ∈ [yc(j-1), yc(j)] (length dmy[j]), z full cell k (length dz[k]).
    for (int k = 1; k <= nz; ++k)
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i) {
                // d(uv)/dx — CV x-faces at xf[i-1] (left) and xf[i] (right) = U positions
                double U_R = 0.5 * (U(i+1,j-1,k) + U(i+1,j,k));     // U at yf[j-1]
                double U_L = 0.5 * (U(i  ,j-1,k) + U(i  ,j,k));
                double v_R = 0.5 * (V(i,j,k) + V(i+1,j,k));         // V at xf[i]
                double v_L = 0.5 * (V(i-1,j,k) + V(i,j,k));
                double flux_uv = (U_R*v_R - U_L*v_L) / dx[i];

                // d(vv)/dy — CV y-faces at yc[j-1] (bot) and yc[j] (top)
                double V_top = 0.5 * (V(i,j,k)   + V(i,j+1,k));     // V at yc[j]
                double V_bot = 0.5 * (V(i,j-1,k) + V(i,j,  k));
                double flux_vv = (V_top*V_top - V_bot*V_bot) / dmy[j];

                // d(wv)/dz — CV z-faces at zf[k-1] (bot) and zf[k] (top)
                double W_top = 0.5 * (W(i,j-1,k+1) + W(i,j,k+1));   // W at yf[j-1]
                double W_bot = 0.5 * (W(i,j-1,k  ) + W(i,j,k  ));
                double v_zt  = 0.5 * (V(i,j,k  ) + V(i,j,k+1));     // V at zf[k]
                double v_zb  = 0.5 * (V(i,j,k-1) + V(i,j,k  ));
                double flux_wv = (W_top*v_zt - W_bot*v_zb) / dz[k];

                Nv(i,j,k) = -(flux_uv + flux_vv + flux_wv);
            }

    // ===== N(w) =================================================
    //  W-CV: x full cell i (length dx[i]), y full cell j (length dy[j]),
    //        z ∈ [zc(k-1), zc(k)] (length dmz[k]).
    for (int k = 1; k <= nz; ++k)
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i) {
                // d(uw)/dx — CV x-faces at xf[i-1] and xf[i]
                double U_R = 0.5 * (U(i+1,j,k-1) + U(i+1,j,k));     // U at zf[k-1]
                double U_L = 0.5 * (U(i  ,j,k-1) + U(i  ,j,k));
                double w_R = 0.5 * (W(i,j,k) + W(i+1,j,k));         // W at xf[i]
                double w_L = 0.5 * (W(i-1,j,k) + W(i,j,k));
                double flux_uw = (U_R*w_R - U_L*w_L) / dx[i];

                // d(vw)/dy — CV y-faces at yf[j-1] and yf[j]
                double V_top = 0.5 * (V(i,j+1,k-1) + V(i,j+1,k));   // V at zf[k-1]
                double V_bot = 0.5 * (V(i,j  ,k-1) + V(i,j  ,k));
                double w_top = 0.5 * (W(i,j,k) + W(i,j+1,k));       // W at yf[j]
                double w_bot = 0.5 * (W(i,j-1,k) + W(i,j,k));
                double flux_vw = (V_top*w_top - V_bot*w_bot) / dy[j];

                // d(ww)/dz — CV z-faces at zc[k-1] (bot) and zc[k] (top)
                double W_zt = 0.5 * (W(i,j,k  ) + W(i,j,k+1));      // W at zc[k]
                double W_zb = 0.5 * (W(i,j,k-1) + W(i,j,k  ));
                double flux_ww = (W_zt*W_zt - W_zb*W_zb) / dmz[k];

                Nw(i,j,k) = -(flux_uw + flux_vw + flux_ww);
            }
}

// ---------------------------------------------------------------------------
//  AB2 update:  q += dt*(1.5*N_new - 0.5*N_old)         (Euler if first step)
// ---------------------------------------------------------------------------
void MomentumSolver::apply_AB2_(Field<double>& U, Field<double>& V, Field<double>& W,
                                const Field<double>& Nu_new, const Field<double>& Nv_new,
                                const Field<double>& Nw_new,
                                const Field<double>& Nu_old, const Field<double>& Nv_old,
                                const Field<double>& Nw_old,
                                double dt, bool first) const
{
    const int nx = nx_, ny = ny_, nz = nz_;
    if (first) {
        for (int k = 1; k <= nz; ++k)
            for (int j = 1; j <= ny; ++j)
                for (int i = 1; i <= nx; ++i) {
                    U(i,j,k) += dt * Nu_new(i,j,k);
                    V(i,j,k) += dt * Nv_new(i,j,k);
                    W(i,j,k) += dt * Nw_new(i,j,k);
                }
    } else {
        for (int k = 1; k <= nz; ++k)
            for (int j = 1; j <= ny; ++j)
                for (int i = 1; i <= nx; ++i) {
                    U(i,j,k) += dt * (1.5*Nu_new(i,j,k) - 0.5*Nu_old(i,j,k));
                    V(i,j,k) += dt * (1.5*Nv_new(i,j,k) - 0.5*Nv_old(i,j,k));
                    W(i,j,k) += dt * (1.5*Nw_new(i,j,k) - 0.5*Nw_old(i,j,k));
                }
    }
}

// ---------------------------------------------------------------------------
//  CN explicit Laplacian half: rhs = q + nu_dt_half * lap(q)
//
//  Component-aware z-stencil: for U/V the z-cell uses dz[k] / dmz[k] / dmz[k+1]
//  (q at zc[k]); for W the z-cell uses dmz[k] / dz[k-1] / dz[k] (q at zf[k-1]).
//  x and y stencils are identical for all three (uniform/periodic in x,y).
// ---------------------------------------------------------------------------
void MomentumSolver::add_explicit_lap_(Component which, const Field<double>& q,
                                       Field<double>& rhs, double nu_dt_half) const
{
    const int nx = nx_, ny = ny_, nz = nz_;
    const auto& dx  = grid_->dx (0); const auto& dmx = grid_->dmx(0);
    const auto& dy  = grid_->dx (1); const auto& dmy = grid_->dmx(1);
    const auto& dz  = grid_->dx (2); const auto& dmz = grid_->dmx(2);

    for (int k = 1; k <= nz; ++k) {
        // z-direction stencil coefficients
        double az, bz, cz;
        if (which == COMP_W) {
            // W at zf[k-1]; CV height dmz[k]; faces at zc[k-1] and zc[k].
            az = 1.0 / (dmz[k] * dz[k-1]);
            cz = 1.0 / (dmz[k] * dz[k  ]);
        } else {
            // U/V at zc[k]; CV height dz[k]; faces at zf[k-1] and zf[k].
            az = 1.0 / (dz[k]  * dmz[k  ]);
            cz = 1.0 / (dz[k]  * dmz[k+1]);
        }
        bz = -(az + cz);

        for (int j = 1; j <= ny; ++j) {
            // y stencil
            double ay, by, cy;
            if (which == COMP_V) {
                ay = 1.0 / (dmy[j] * dy[j-1]);
                cy = 1.0 / (dmy[j] * dy[j  ]);
            } else {
                ay = 1.0 / (dy[j]  * dmy[j  ]);
                cy = 1.0 / (dy[j]  * dmy[j+1]);
            }
            by = -(ay + cy);

            for (int i = 1; i <= nx; ++i) {
                // x stencil
                double ax, bx, cx;
                if (which == COMP_U) {
                    ax = 1.0 / (dmx[i] * dx[i-1]);
                    cx = 1.0 / (dmx[i] * dx[i  ]);
                } else {
                    ax = 1.0 / (dx[i]  * dmx[i  ]);
                    cx = 1.0 / (dx[i]  * dmx[i+1]);
                }
                bx = -(ax + cx);

                double lap = ax*q(i-1,j,k) + bx*q(i,j,k) + cx*q(i+1,j,k)
                           + ay*q(i,j-1,k) + by*q(i,j,k) + cy*q(i,j+1,k)
                           + az*q(i,j,k-1) + bz*q(i,j,k) + cz*q(i,j,k+1);

                rhs(i,j,k) = q(i,j,k) + nu_dt_half * lap;
            }
        }
    }
}

// ---------------------------------------------------------------------------
//  ADI z-sweep: solve (I - nu_dt_half * d²/dz²) q = q_in (rhs lives in q)
//  Wall BC z (low/high rank only):
//    U, V: antisymmetric ghost — fold into diagonal (A=0, B += a_z; C=0, B += c_z)
//    W   : ghost = 0 at wall face — at low rank, pin row k=1 to RHS=0;
//          at top rank, just zero C (top ghost is pinned wall = 0).
// ---------------------------------------------------------------------------
void MomentumSolver::adi_z_(Component which, Field<double>& q, double nu_dt_half)
{
    const int nx = nx_, ny = ny_, nz = nz_;
    const auto& dz  = grid_->dx (2);
    const auto& dmz = grid_->dmx(2);
    const int ns = nx * ny;

    const bool low_wall  = (rank_z_ == 0);
    const bool high_wall = (rank_z_ == np3_ - 1);
    const bool is_w      = (which == COMP_W);

    for (int k = 1; k <= nz; ++k) {
        double az_raw, cz_raw;
        if (is_w) {
            az_raw = 1.0 / (dmz[k] * dz[k-1]);
            cz_raw = 1.0 / (dmz[k] * dz[k  ]);
        } else {
            az_raw = 1.0 / (dz[k]  * dmz[k  ]);
            cz_raw = 1.0 / (dz[k]  * dmz[k+1]);
        }
        const double az = -nu_dt_half * az_raw;   // off-diagonal (negative)
        const double cz = -nu_dt_half * cz_raw;
        const double bz = 1.0 - az - cz;          // = 1 + nu_dt_half*(az_raw + cz_raw)

        const bool is_bot = (low_wall  && k == 1 );
        const bool is_top = (high_wall && k == nz);

        // Wall-row modifications
        double Ak = az, Bk = bz, Ck = cz;
        if (is_bot) {
            Ak = 0.0;
            if (!is_w) Bk -= az;   // antisymm fold for U/V (az is negative → Bk += |az|)
        }
        if (is_top) {
            Ck = 0.0;
            if (!is_w) Bk -= cz;
        }

        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i) {
                int row = k - 1;
                int s   = (i-1)*ny + (j-1);
                Az_[row*ns + s] = Ak;
                Bz_[row*ns + s] = Bk;
                Cz_[row*ns + s] = Ck;
                Dz_[row*ns + s] = q(i,j,k);
            }
    }

    // W bottom-wall row: pin to D=0 (wall face must remain 0)
    if (low_wall && is_w) {
        for (int s = 0; s < ns; ++s) {
            Az_[0*ns + s] = 0.0;
            Bz_[0*ns + s] = 1.0;
            Cz_[0*ns + s] = 0.0;
            Dz_[0*ns + s] = 0.0;
        }
    }

    {
        double t0 = MPI_Wtime();
        fdma_z_->set_rho(Az_.data(), Bz_.data(), Cz_.data(), ns);
        fdma_z_->solve(Az_.data(), Bz_.data(), Cz_.data(), Dz_.data());
        if (step_count_ > 20000) tdma_time_ += MPI_Wtime() - t0;
    }

    for (int k = 1; k <= nz; ++k)
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i)
                q(i,j,k) = Dz_[(k-1)*ns + (i-1)*ny + (j-1)];
}

// ---------------------------------------------------------------------------
//  ADI y-sweep: (I - nu_dt_half * d²/dy²) q = q_in   (uniform y, periodic)
// ---------------------------------------------------------------------------
void MomentumSolver::adi_y_(Component which, Field<double>& q, double nu_dt_half)
{
    const int nx = nx_, ny = ny_, nz = nz_;
    const auto& dy  = grid_->dx (1); const auto& dmy = grid_->dmx(1);
    const int ns = nx * nz;

    for (int j = 1; j <= ny; ++j) {
        double ay_raw, cy_raw;
        if (which == COMP_V) {
            ay_raw = 1.0 / (dmy[j] * dy[j-1]);
            cy_raw = 1.0 / (dmy[j] * dy[j  ]);
        } else {
            ay_raw = 1.0 / (dy[j]  * dmy[j  ]);
            cy_raw = 1.0 / (dy[j]  * dmy[j+1]);
        }
        const double ay = -nu_dt_half * ay_raw;
        const double cy = -nu_dt_half * cy_raw;
        const double by = 1.0 - ay - cy;

        for (int k = 1; k <= nz; ++k)
            for (int i = 1; i <= nx; ++i) {
                int row = j - 1;
                int s   = (i-1)*nz + (k-1);
                Ay_[row*ns + s] = ay;
                By_[row*ns + s] = by;
                Cy_[row*ns + s] = cy;
                Dy_[row*ns + s] = q(i,j,k);
            }
    }

    {
        double t0 = MPI_Wtime();
        fdma_y_->set_rho(Ay_.data(), By_.data(), Cy_.data(), ns);
        if (cfg_->pbc2)
            fdma_y_->solve_cycl(Ay_.data(), By_.data(), Cy_.data(), Dy_.data());
        else
            fdma_y_->solve(Ay_.data(), By_.data(), Cy_.data(), Dy_.data());
        if (step_count_ > 20000) tdma_time_ += MPI_Wtime() - t0;
    }

    for (int k = 1; k <= nz; ++k)
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i)
                q(i,j,k) = Dy_[(j-1)*ns + (i-1)*nz + (k-1)];
}

// ---------------------------------------------------------------------------
//  ADI x-sweep: (I - nu_dt_half * d²/dx²) q = q_in  (uniform x, periodic)
// ---------------------------------------------------------------------------
void MomentumSolver::adi_x_(Component which, Field<double>& q, double nu_dt_half,
                            Field<double>* dst)
{
    Field<double>& out = dst ? *dst : q;
    const int nx = nx_, ny = ny_, nz = nz_;
    const auto& dx  = grid_->dx (0); const auto& dmx = grid_->dmx(0);
    const int ns = ny * nz;

    for (int i = 1; i <= nx; ++i) {
        double ax_raw, cx_raw;
        if (which == COMP_U) {
            ax_raw = 1.0 / (dmx[i] * dx[i-1]);
            cx_raw = 1.0 / (dmx[i] * dx[i  ]);
        } else {
            ax_raw = 1.0 / (dx[i]  * dmx[i  ]);
            cx_raw = 1.0 / (dx[i]  * dmx[i+1]);
        }
        const double ax = -nu_dt_half * ax_raw;
        const double cx = -nu_dt_half * cx_raw;
        const double bx = 1.0 - ax - cx;

        for (int k = 1; k <= nz; ++k)
            for (int j = 1; j <= ny; ++j) {
                int row = i - 1;
                int s   = (j-1)*nz + (k-1);
                Ax_[row*ns + s] = ax;
                Bx_[row*ns + s] = bx;
                Cx_[row*ns + s] = cx;
                Dx_[row*ns + s] = q(i,j,k);
            }
    }

    {
        double t0 = MPI_Wtime();
        fdma_x_->set_rho(Ax_.data(), Bx_.data(), Cx_.data(), ns);
        if (cfg_->pbc1)
            fdma_x_->solve_cycl(Ax_.data(), Bx_.data(), Cx_.data(), Dx_.data());
        else
            fdma_x_->solve(Ax_.data(), Bx_.data(), Cx_.data(), Dx_.data());
        if (step_count_ > 20000) tdma_time_ += MPI_Wtime() - t0;
    }

    for (int k = 1; k <= nz; ++k)
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i)
                out(i,j,k) = Dx_[(i-1)*ns + (j-1)*nz + (k-1)];
}

// ---------------------------------------------------------------------------
//  advance — one momentum predictor step (advection + AB2 + body force + ADI heat)
//
//  P (current pressure) is unused here: this is Chorin-style projection in which
//  the FULL pressure increment is computed by PressureSolver downstream and added
//  to the predicted velocity via velocity correction.
// ---------------------------------------------------------------------------
void MomentumSolver::advance(Field<double>& U, Field<double>& V, Field<double>& W,
                             const Field<double>& /*P*/,
                             double dt, double mean_dPdx)
{
    const int nx = nx_, ny = ny_, nz = nz_;

    ++step_count_;
    const double t_adv0 = MPI_Wtime();

    // Refresh FilteredTDMA truncation threshold to reflect the current dt
    // (eps_ = dt / (N*N) inside the library — see filtered_tdma.hpp).
    fdma_x_->set_eps_constant(dt);
    fdma_y_->set_eps_constant(dt);
    fdma_z_->set_eps_constant(dt);

    // === Step 1a: advection N at u^n (needs current ghosts; assumed already set) ===
    compute_advection_(U, V, W, Nu_new_, Nv_new_, Nw_new_);

    // === Step 1b: AB2 explicit time advance of advection ===
    apply_AB2_(U, V, W, Nu_new_, Nv_new_, Nw_new_,
               Nu_old_, Nv_old_, Nw_old_, dt, first_step_);

    // === Step 1c: body force on U (channel streamwise pressure gradient) ===
    for (int k = 1; k <= nz; ++k)
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i)
                U(i,j,k) += dt * (-mean_dPdx);

    // === Refresh halos and walls before diffusion (lap stencil reads ghosts) ===
    halo_->exchange(U); halo_->exchange(V); halo_->exchange(W);
    // Apply z-wall BC inline (antisymm for U/V, zero for W) — matches v2 boundary.cpp.
    if (rank_z_ == 0) {
        for (int j = 0; j <= ny + 1; ++j)
            for (int i = 0; i <= nx + 1; ++i) {
                U(i,j,0) = -U(i,j,1);    // antisymm
                V(i,j,0) = -V(i,j,1);
                W(i,j,1) =  0.0;          // bottom wall face
            }
    }
    if (rank_z_ == np3_ - 1) {
        for (int j = 0; j <= ny + 1; ++j)
            for (int i = 0; i <= nx + 1; ++i) {
                U(i,j,nz+1) = -U(i,j,nz);
                V(i,j,nz+1) = -V(i,j,nz);
                W(i,j,nz+1) =  0.0;       // top wall face
            }
    }

    // === Step 1d: CN-ADI diffusion per component ===
    const double nu       = inv_Re_;
    const double nu_dt_h  = 0.5 * nu * dt;

    auto diffuse = [&](Component c, Field<double>& q) {
        // rhs = q + nu*dt/2 * lap(q)
        add_explicit_lap_(c, q, rhs_, nu_dt_h);
        // (I - nu*dt/2 * d²/dz²)(I - .. d²/dy²)(I - .. d²/dx²) q = rhs
        adi_z_(c, rhs_, nu_dt_h);
        adi_y_(c, rhs_, nu_dt_h);
        // Final x-sweep unpacks straight into q — saves a full-field copy.
        adi_x_(c, rhs_, nu_dt_h, &q);
    };

    diffuse(COMP_U, U);
    diffuse(COMP_V, V);
    diffuse(COMP_W, W);

    // === Rotate AB2 history ===
    // Swap buffers instead of copying: next step's compute_advection_ overwrites
    // *_new_ entirely, so the old contents (now in *_new_) are irrelevant.
    Nu_old_.swap(Nu_new_);
    Nv_old_.swap(Nv_new_);
    Nw_old_.swap(Nw_new_);
    first_step_ = false;

    if (step_count_ > 20000) momentum_time_ += MPI_Wtime() - t_adv0;
}

} // namespace channel
