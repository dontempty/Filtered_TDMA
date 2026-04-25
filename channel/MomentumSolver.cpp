// channel/MomentumSolver.cpp — Beam-Warming 3-stage ADI momentum solver.
//
// Matches PaScaL_TCS module_solve_momentum.f90 structure:
//   Stage 1 (x): periodic  — FilteredTdmaSolver::solve_cycl
//   Stage 2 (y): periodic  — FilteredTdmaSolver::solve_cycl
//   Stage 3 (z): wall      — FilteredTdmaSolver::solve
//
// Coefficient convention (PaScaL_TCS analogue, constant ρ/μ simplification):
//   nu_h = 0.5 / Re_b   (invRhocCmu_half with Mu=1, invRhoc=1)
//
//   mAM  = -nu_h/(cv*sp_minus) + 0.25*(-u_face_m/sp_minus + 0.5*dq/dx_minus)
//   mAP  = -nu_h/(cv*sp_plus)  + 0.25*( u_face_p/sp_plus  + 0.5*dq/dx_plus)
//   mAC  =  nu_h/cv*(1/sp_m+1/sp_p) + 0.25*(...)
//
//   where cv = control-volume width, sp = neighbor spacing,
//         u_face = advecting velocity at CV face (interpolated to Q position).
//
// RHS = dt * (ν∇²q  −  ∇p  +  forcing  −  M_all·q^n)
//   No explicit convection: all convection enters via the Beam-Warming matrix.
//
// Wall BC in z-sweep: bottom rank pins first TDMA row to dQ=0 (no-slip / no-penetration);
//   top rank zeroes the upper off-diagonal of the last row.

#include "MomentumSolver.hpp"

#include "Config.hpp"
#include "Grid.hpp"
#include "HaloExchanger.hpp"
#include "MpiTopology.hpp"
#include "Subdomain.hpp"

#include <algorithm>
#include <cmath>
#include <mpi.h>

namespace channel {

MomentumSolver::MomentumSolver(const Config& cfg,
                               const MpiTopology& topo,
                               const Subdomain& sub,
                               const Grid& grid,
                               const HaloExchanger& halo)
    : cfg_(&cfg), sub_(&sub), grid_(&grid), halo_(&halo),
      dU_(sub.nx(), sub.ny(), sub.nz()),
      dV_(sub.nx(), sub.ny(), sub.nz()),
      dW_(sub.nx(), sub.ny(), sub.nz())
{
    inv_Re_ = 1.0 / cfg.Re_b;
    np3_    = topo.dim(2);
    rank_z_ = topo.rank_in(2);

    nx_ = sub.nx(); ny_ = sub.ny(); nz_ = sub.nz();

    const auto backend = TdmaSolver::parse_backend(cfg.tdma_backend);
    fdma_x_ = std::make_unique<TdmaSolver>(topo, 0, ny_*nz_, nx_, cfg.pbc1, backend);
    fdma_y_ = std::make_unique<TdmaSolver>(topo, 1, nx_*nz_, ny_, cfg.pbc2, backend);
    fdma_z_ = std::make_unique<TdmaSolver>(topo, 2, nx_*ny_, nz_, false,    backend);

    Ax_.resize((std::size_t)nx_ * ny_*nz_);
    Bx_.resize((std::size_t)nx_ * ny_*nz_);
    Cx_.resize((std::size_t)nx_ * ny_*nz_);
    Dx_.resize((std::size_t)nx_ * ny_*nz_);

    Ay_.resize((std::size_t)ny_ * nx_*nz_);
    By_.resize((std::size_t)ny_ * nx_*nz_);
    Cy_.resize((std::size_t)ny_ * nx_*nz_);
    Dy_.resize((std::size_t)ny_ * nx_*nz_);

    Az_.resize((std::size_t)nz_ * nx_*ny_);
    Bz_.resize((std::size_t)nz_ * nx_*ny_);
    Cz_.resize((std::size_t)nz_ * nx_*ny_);
    Dz_.resize((std::size_t)nz_ * nx_*ny_);
}

// ---------------------------------------------------------------------------
// compute_rhs_
//
// Sets dQ = dt * (ν∇²q  −  ∇p  +  forcing  −  M_all·q^n)
//
// Grid spacing conventions:
//   dx[i]  = x cell width (face-to-face)   dmx[i] = x center-to-center
//   For uniform x/y: dx == dmx == h (same value for all i/j)
//
//   U lives at x-faces → CV width in x = dmx[i], spacing = dx[i-1]/dx[i]
//   V lives at y-faces → CV width in y = dmy[j], spacing = dy[j-1]/dy[j]
//   W lives at z-faces → CV width in z = dmz[k], spacing = dz[k-1]/dz[k]
//   All others (e.g. U in y): CV width = dy[j], spacing = dmy[j]/dmy[j+1]
// ---------------------------------------------------------------------------
void MomentumSolver::compute_rhs_(Component which,
                                  Field<double>& dQ,
                                  const Field<double>& U,
                                  const Field<double>& V,
                                  const Field<double>& W,
                                  const Field<double>& P,
                                  double dt, double mean_dPdx)
{
    const int nx = nx_, ny = ny_, nz = nz_;
    const auto& dx  = grid_->dx (0);  const auto& dmx = grid_->dmx(0);
    const auto& dy  = grid_->dx (1);  const auto& dmy = grid_->dmx(1);
    const auto& dz  = grid_->dx (2);  const auto& dmz = grid_->dmx(2);
    const double nu   = inv_Re_;
    const double nu_h = 0.5 * nu;

    auto qref = [&](int i, int j, int k) -> double {
        if (which == COMP_U) return U(i,j,k);
        if (which == COMP_V) return V(i,j,k);
        return W(i,j,k);
    };

    // Beam-Warming: keep the 0.5·∂q/∂d "self-derivative" coefficient ONLY in
    // the direction `d` whose convecting velocity equals the solved component
    // (PaScaL_TCS convention).  In cross directions, the convecting velocity
    // is a different field, and the 0.5·∂q/∂d contribution is non-physical
    // — it adds a spurious u·∂u/∂z-type term to U eq even when W = 0, which
    // breaks Poiseuille equilibrium.
    const double sx = (which == COMP_U) ? 1.0 : 0.0;
    const double sy = (which == COMP_V) ? 1.0 : 0.0;
    const double sz = (which == COMP_W) ? 1.0 : 0.0;

    for (int k = 1; k <= nz; ++k)
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i) {

                const double q   = qref(i,j,k);
                const double qxm = qref(i-1,j,k);
                const double qxp = qref(i+1,j,k);
                const double qym = qref(i,j-1,k);
                const double qyp = qref(i,j+1,k);
                const double qzm = qref(i,j,k-1);
                const double qzp = qref(i,j,k+1);

                // ==============================================================
                // x-direction BW coefficients
                // For U: cv=dmx[i], sp_m=dx[i-1], sp_p=dx[i]  (U at x-face)
                // For V,W: cv=dx[i], sp_m=dmx[i], sp_p=dmx[i+1] (at x-cell)
                // Uniform x: dx==dmx so these are identical.
                // ==============================================================
                double cvx, spxm, spxp;
                if (which == COMP_U) {
                    cvx = dmx[i]; spxm = dx[i-1]; spxp = dx[i];
                } else {
                    cvx = dx[i]; spxm = dmx[i]; spxp = dmx[i+1];
                }
                double dqdx1 = (q - qxm) / spxm;
                double dqdx2 = (qxp - q) / spxp;
                double ux1, ux2;
                if (which == COMP_U) {
                    ux1 = 0.5*(U(i-1,j,k) + U(i,j,k));
                    ux2 = 0.5*(U(i,j,k)   + U(i+1,j,k));
                } else if (which == COMP_V) {
                    ux1 = 0.5*(U(i,j-1,k)   + U(i,j,k));      // y-interp
                    ux2 = 0.5*(U(i+1,j-1,k) + U(i+1,j,k));
                } else {
                    ux1 = 0.5*(U(i,j,k-1)   + U(i,j,k));      // z-interp
                    ux2 = 0.5*(U(i+1,j,k-1) + U(i+1,j,k));
                }
                double mAMI = -nu_h/(cvx*spxm) + 0.25*(-ux1/spxm + sx*0.5*dqdx1);
                double mAPI = -nu_h/(cvx*spxp)  + 0.25*( ux2/spxp + sx*0.5*dqdx2);
                double mACI =  nu_h/cvx*(1.0/spxm + 1.0/spxp)
                             + 0.25*(sx*(0.5*dqdx2 + 0.5*dqdx1) - ux2/spxp + ux1/spxm);

                // ==============================================================
                // y-direction BW coefficients
                // For V: cv=dmy[j], sp_m=dy[j-1], sp_p=dy[j]
                // For U,W: cv=dy[j], sp_m=dmy[j], sp_p=dmy[j+1]
                // ==============================================================
                double cvy, spym, spyp;
                if (which == COMP_V) {
                    cvy = dmy[j]; spym = dy[j-1]; spyp = dy[j];
                } else {
                    cvy = dy[j]; spym = dmy[j]; spyp = dmy[j+1];
                }
                double dqdy1 = (q - qym) / spym;
                double dqdy2 = (qyp - q) / spyp;
                double vy1, vy2;
                if (which == COMP_V) {
                    vy1 = 0.5*(V(i,j-1,k) + V(i,j,k));
                    vy2 = 0.5*(V(i,j,k)   + V(i,j+1,k));
                } else if (which == COMP_U) {
                    vy1 = 0.5*(V(i-1,j,k)   + V(i,j,k));      // x-interp
                    vy2 = 0.5*(V(i-1,j+1,k) + V(i,j+1,k));
                } else {
                    vy1 = 0.5*(V(i,j,k-1)   + V(i,j,k));      // z-interp
                    vy2 = 0.5*(V(i,j+1,k-1) + V(i,j+1,k));
                }
                double mAMJ = -nu_h/(cvy*spym) + 0.25*(-vy1/spym + sy*0.5*dqdy1);
                double mAPJ = -nu_h/(cvy*spyp)  + 0.25*( vy2/spyp + sy*0.5*dqdy2);
                double mACJ =  nu_h/cvy*(1.0/spym + 1.0/spyp)
                             + 0.25*(sy*(0.5*dqdy2 + 0.5*dqdy1) - vy2/spyp + vy1/spym);

                // ==============================================================
                // z-direction BW coefficients (non-uniform z)
                // For W: cv=dmz[k], sp_m=dz[k-1], sp_p=dz[k]  (W at z-face)
                // For U,V: cv=dz[k], sp_m=dmz[k], sp_p=dmz[k+1] (at z-cell)
                // ==============================================================
                double cvz, spzm, spzp;
                if (which == COMP_W) {
                    cvz = dmz[k]; spzm = dz[k-1]; spzp = dz[k];
                } else {
                    cvz = dz[k]; spzm = dmz[k]; spzp = dmz[k+1];
                }
                double dqdz1 = (q - qzm) / spzm;
                double dqdz2 = (qzp - q) / spzp;
                double wz1, wz2;
                if (which == COMP_W) {
                    wz1 = 0.5*(W(i,j,k-1) + W(i,j,k));
                    wz2 = 0.5*(W(i,j,k)   + W(i,j,k+1));
                } else if (which == COMP_U) {
                    wz1 = 0.5*(W(i-1,j,k)   + W(i,j,k));      // x-interp
                    wz2 = 0.5*(W(i-1,j,k+1) + W(i,j,k+1));
                } else {
                    wz1 = 0.5*(W(i,j-1,k)   + W(i,j,k));      // y-interp
                    wz2 = 0.5*(W(i,j-1,k+1) + W(i,j,k+1));
                }
                double mAMK = -nu_h/(cvz*spzm) + 0.25*(-wz1/spzm + sz*0.5*dqdz1);
                double mAPK = -nu_h/(cvz*spzp)  + 0.25*( wz2/spzp + sz*0.5*dqdz2);
                double mACK =  nu_h/cvz*(1.0/spzm + 1.0/spzp)
                             + 0.25*(sz*(0.5*dqdz2 + 0.5*dqdz1) - wz2/spzp + wz1/spzm);

                // ==============================================================
                // Explicit viscous (½)·ν∇²q.  The factor ½ matches PaScaL_TCS
                // line 341 (invRhocCmu_half · ...).  After subtracting imp_res
                // = M_all·u^n = -½·A·u + N(u^n)  (quadratic N), the net RHS
                // becomes dt·(A·u − N(u) + force − ∇p), consistent with full
                // Crank–Nicolson + Beam–Warming.
                // ==============================================================
                double diff_x = nu_h * ((qxp - q)/spxp - (q - qxm)/spxm) / cvx;
                double diff_y = nu_h * ((qyp - q)/spyp - (q - qym)/spym) / cvy;
                double diff_z = nu_h * ((qzp - q)/spzp - (q - qzm)/spzm) / cvz;

                // ==============================================================
                // Pressure gradient
                // ==============================================================
                double press = 0.0, force = 0.0;
                if (which == COMP_U) {
                    press = (P(i,j,k) - P(i-1,j,k)) / dmx[i];
                    force = -mean_dPdx;
                } else if (which == COMP_V) {
                    press = (P(i,j,k) - P(i,j-1,k)) / dmy[j];
                } else {
                    // W at z-face k-1/2; skip when k==1 on bottom rank (ghost)
                    if (k > 1 || rank_z_ > 0)
                        press = (P(i,j,k) - P(i,j,k-1)) / dmz[k];
                }

                // ==============================================================
                // Implicit residual: M_all·q^n
                //   M_uu·q only captures the diagonal block of N'(u^n).
                //   For the U eq:  (∂N_U/∂U)·U = 2·N_x + N_y + N_z
                //   The cross blocks (∂N_U/∂V)·V = N_y_cross  and
                //                    (∂N_U/∂W)·W = N_z_cross  are NOT captured.
                //   Add them explicitly so 0.5·N'·u^n = N(u^n) holds (CN form),
                //   matching MPM-STD's M12Vn / M13Wn subtractions
                //   (core_momentum.f90:877-883).
                // ==============================================================
                double imp_res = (mACI + mACJ + mACK) * q
                               + mAPI * qxp + mAMI * qxm
                               + mAPJ * qyp + mAMJ * qym
                               + mAPK * qzp + mAMK * qzm;

                // Cross-block contributions (CN factor 0.5, face-avg factor 0.5 → 0.25)
                if (which == COMP_U) {
                    imp_res += 0.25 * (vy1*dqdy1 + vy2*dqdy2);   // 0.5·N_y cross
                    imp_res += 0.25 * (wz1*dqdz1 + wz2*dqdz2);   // 0.5·N_z cross
                } else if (which == COMP_V) {
                    imp_res += 0.25 * (ux1*dqdx1 + ux2*dqdx2);   // 0.5·N_x cross
                    imp_res += 0.25 * (wz1*dqdz1 + wz2*dqdz2);   // 0.5·N_z cross
                } else { // COMP_W
                    imp_res += 0.25 * (ux1*dqdx1 + ux2*dqdx2);   // 0.5·N_x cross
                    imp_res += 0.25 * (vy1*dqdy1 + vy2*dqdy2);   // 0.5·N_y cross
                }

                dQ(i,j,k) = dt * (diff_x + diff_y + diff_z - press + force - imp_res);
            }
}

// ---------------------------------------------------------------------------
// adi_sweep_x_  — solve (I + dt·Mx)·dQ = dQ_in  in place
// TDMA layout [nx × (ny*nz)], row=i-1, sys=(j-1)*nz+(k-1)
// ---------------------------------------------------------------------------
void MomentumSolver::adi_sweep_x_(Component which, Field<double>& dQ, double dt,
                                   const Field<double>& U, const Field<double>& V,
                                   const Field<double>& W)
{
    const int nx = nx_, ny = ny_, nz = nz_;
    const auto& dx  = grid_->dx (0);
    const auto& dmx = grid_->dmx(0);
    const double nu_h = 0.5 * inv_Re_;
    const int ns = ny * nz;
    const double sx = (which == COMP_U) ? 1.0 : 0.0;   // BW self-derivative gate

    for (int k = 1; k <= nz; ++k)
        for (int j = 1; j <= ny; ++j) {
            const int s = (j-1)*nz + (k-1);
            for (int i = 1; i <= nx; ++i) {
                const int row = i-1;

                double cvx, spxm, spxp;
                if (which == COMP_U) {
                    cvx = dmx[i]; spxm = dx[i-1]; spxp = dx[i];
                } else {
                    cvx = dx[i]; spxm = dmx[i]; spxp = dmx[i+1];
                }

                // Q at x-neighbors (original velocity, not dQ)
                double qxm, qx, qxp;
                if (which == COMP_U) { qxm=U(i-1,j,k); qx=U(i,j,k); qxp=U(i+1,j,k); }
                else if (which == COMP_V) { qxm=V(i-1,j,k); qx=V(i,j,k); qxp=V(i+1,j,k); }
                else { qxm=W(i-1,j,k); qx=W(i,j,k); qxp=W(i+1,j,k); }

                double dqdx1 = (qx - qxm) / spxm;
                double dqdx2 = (qxp - qx) / spxp;

                double ux1, ux2;
                if (which == COMP_U) {
                    ux1 = 0.5*(U(i-1,j,k) + U(i,j,k));
                    ux2 = 0.5*(U(i,j,k)   + U(i+1,j,k));
                } else if (which == COMP_V) {
                    ux1 = 0.5*(U(i,j-1,k)   + U(i,j,k));
                    ux2 = 0.5*(U(i+1,j-1,k) + U(i+1,j,k));
                } else {
                    ux1 = 0.5*(U(i,j,k-1)   + U(i,j,k));
                    ux2 = 0.5*(U(i+1,j,k-1) + U(i+1,j,k));
                }

                double mAMI = -nu_h/(cvx*spxm) + 0.25*(-ux1/spxm + sx*0.5*dqdx1);
                double mAPI = -nu_h/(cvx*spxp)  + 0.25*( ux2/spxp + sx*0.5*dqdx2);
                double mACI =  nu_h/cvx*(1.0/spxm + 1.0/spxp)
                             + 0.25*(sx*(0.5*dqdx2 + 0.5*dqdx1) - ux2/spxp + ux1/spxm);

                Ax_[row*ns + s] = mAMI * dt;
                Cx_[row*ns + s] = mAPI * dt;
                Bx_[row*ns + s] = mACI * dt + 1.0;
                Dx_[row*ns + s] = dQ(i,j,k);
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
        for (int j = 1; j <= ny; ++j) {
            const int s = (j-1)*nz + (k-1);
            for (int i = 1; i <= nx; ++i)
                dQ(i,j,k) = Dx_[(i-1)*ns + s];
        }
}

// ---------------------------------------------------------------------------
// adi_sweep_y_  — solve (I + dt·My)·dQ = dQ_in  in place
// TDMA layout [ny × (nx*nz)], row=j-1, sys=(i-1)*nz+(k-1)
// ---------------------------------------------------------------------------
void MomentumSolver::adi_sweep_y_(Component which, Field<double>& dQ, double dt,
                                   const Field<double>& U, const Field<double>& V,
                                   const Field<double>& W)
{
    const int nx = nx_, ny = ny_, nz = nz_;
    const auto& dy  = grid_->dx (1);
    const auto& dmy = grid_->dmx(1);
    const double nu_h = 0.5 * inv_Re_;
    const int ns = nx * nz;
    const double sy = (which == COMP_V) ? 1.0 : 0.0;   // BW self-derivative gate

    for (int k = 1; k <= nz; ++k)
        for (int i = 1; i <= nx; ++i) {
            const int s = (i-1)*nz + (k-1);
            for (int j = 1; j <= ny; ++j) {
                const int row = j-1;

                double cvy, spym, spyp;
                if (which == COMP_V) {
                    cvy = dmy[j]; spym = dy[j-1]; spyp = dy[j];
                } else {
                    cvy = dy[j]; spym = dmy[j]; spyp = dmy[j+1];
                }

                double qym, qy, qyp;
                if (which == COMP_U) { qym=U(i,j-1,k); qy=U(i,j,k); qyp=U(i,j+1,k); }
                else if (which == COMP_V) { qym=V(i,j-1,k); qy=V(i,j,k); qyp=V(i,j+1,k); }
                else { qym=W(i,j-1,k); qy=W(i,j,k); qyp=W(i,j+1,k); }

                double dqdy1 = (qy - qym) / spym;
                double dqdy2 = (qyp - qy) / spyp;

                double vy1, vy2;
                if (which == COMP_V) {
                    vy1 = 0.5*(V(i,j-1,k) + V(i,j,k));
                    vy2 = 0.5*(V(i,j,k)   + V(i,j+1,k));
                } else if (which == COMP_U) {
                    vy1 = 0.5*(V(i-1,j,k)   + V(i,j,k));
                    vy2 = 0.5*(V(i-1,j+1,k) + V(i,j+1,k));
                } else {
                    vy1 = 0.5*(V(i,j,k-1)   + V(i,j,k));
                    vy2 = 0.5*(V(i,j+1,k-1) + V(i,j+1,k));
                }

                double mAMJ = -nu_h/(cvy*spym) + 0.25*(-vy1/spym + sy*0.5*dqdy1);
                double mAPJ = -nu_h/(cvy*spyp)  + 0.25*( vy2/spyp + sy*0.5*dqdy2);
                double mACJ =  nu_h/cvy*(1.0/spym + 1.0/spyp)
                             + 0.25*(sy*(0.5*dqdy2 + 0.5*dqdy1) - vy2/spyp + vy1/spym);

                Ay_[row*ns + s] = mAMJ * dt;
                Cy_[row*ns + s] = mAPJ * dt;
                By_[row*ns + s] = mACJ * dt + 1.0;
                Dy_[row*ns + s] = dQ(i,j,k);
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
        for (int i = 1; i <= nx; ++i) {
            const int s = (i-1)*nz + (k-1);
            for (int j = 1; j <= ny; ++j)
                dQ(i,j,k) = Dy_[(j-1)*ns + s];
        }
}

// ---------------------------------------------------------------------------
// adi_sweep_z_  — solve (I + dt·Mz)·dQ = dQ_in  in place
// TDMA layout [nz × (nx*ny)], row=k-1, sys=(i-1)*ny+(j-1)
// Wall BCs: bottom rank pins row 0 (k=1) → dQ=0; top rank zero ap at last row.
// ---------------------------------------------------------------------------
void MomentumSolver::adi_sweep_z_(Component which, Field<double>& dQ, double dt,
                                   const Field<double>& U, const Field<double>& V,
                                   const Field<double>& W)
{
    const int nx = nx_, ny = ny_, nz = nz_;
    const auto& dz  = grid_->dx (2);
    const auto& dmz = grid_->dmx(2);
    const double nu_h = 0.5 * inv_Re_;
    const int ns = nx * ny;
    const double sz = (which == COMP_W) ? 1.0 : 0.0;   // BW self-derivative gate

    for (int j = 1; j <= ny; ++j)
        for (int i = 1; i <= nx; ++i) {
            const int s = (i-1)*ny + (j-1);
            for (int k = 1; k <= nz; ++k) {
                const int row = k-1;

                // W at z-face: cv=dmz[k], sp=dz[k-1]/dz[k]
                // U/V at z-cell: cv=dz[k], sp=dmz[k]/dmz[k+1]
                double cvz, spzm, spzp;
                if (which == COMP_W) {
                    cvz = dmz[k]; spzm = dz[k-1]; spzp = dz[k];
                } else {
                    cvz = dz[k]; spzm = dmz[k]; spzp = dmz[k+1];
                }

                double qzm, qz, qzp;
                if (which == COMP_U) { qzm=U(i,j,k-1); qz=U(i,j,k); qzp=U(i,j,k+1); }
                else if (which == COMP_V) { qzm=V(i,j,k-1); qz=V(i,j,k); qzp=V(i,j,k+1); }
                else { qzm=W(i,j,k-1); qz=W(i,j,k); qzp=W(i,j,k+1); }

                double dqdz1 = (qz - qzm) / spzm;
                double dqdz2 = (qzp - qz) / spzp;

                double wz1, wz2;
                if (which == COMP_W) {
                    wz1 = 0.5*(W(i,j,k-1) + W(i,j,k));
                    wz2 = 0.5*(W(i,j,k)   + W(i,j,k+1));
                } else if (which == COMP_U) {
                    wz1 = 0.5*(W(i-1,j,k)   + W(i,j,k));
                    wz2 = 0.5*(W(i-1,j,k+1) + W(i,j,k+1));
                } else {
                    wz1 = 0.5*(W(i,j-1,k)   + W(i,j,k));
                    wz2 = 0.5*(W(i,j-1,k+1) + W(i,j,k+1));
                }

                double mAMK = -nu_h/(cvz*spzm) + 0.25*(-wz1/spzm + sz*0.5*dqdz1);
                double mAPK = -nu_h/(cvz*spzp)  + 0.25*( wz2/spzp + sz*0.5*dqdz2);
                double mACK =  nu_h/cvz*(1.0/spzm + 1.0/spzp)
                             + 0.25*(sz*(0.5*dqdz2 + 0.5*dqdz1) - wz2/spzp + wz1/spzm);

                Az_[row*ns + s] = mAMK * dt;
                Cz_[row*ns + s] = mAPK * dt;
                Bz_[row*ns + s] = mACK * dt + 1.0;
                Dz_[row*ns + s] = dQ(i,j,k);
            }
        }

    // Wall BCs for the z-sweep:
    //   W (z-face):  W(k=1) is the bottom wall face (=0); W(k=nz+1) is the top
    //                wall face (=0). Pin first row at bottom; zero ap at top row.
    //   U,V (cell):  antisymmetric ghost U(k=0) = -U(k=1), U(k=nz+1) = -U(k=nz).
    //                Fold the am·U(0) = -am·U(1) into ac (ac -= am), then am = 0
    //                at bottom. Same for ap at top.
    if (rank_z_ == 0) {
        if (which == COMP_W) {
            for (int s = 0; s < ns; ++s) {
                Az_[0*ns+s] = 0.0;  Cz_[0*ns+s] = 0.0;
                Bz_[0*ns+s] = 1.0;  Dz_[0*ns+s] = 0.0;
            }
        } else {
            for (int s = 0; s < ns; ++s) {
                Bz_[0*ns+s] -= Az_[0*ns+s];
                Az_[0*ns+s]  = 0.0;
            }
        }
    }
    if (rank_z_ == np3_-1) {
        if (which == COMP_W) {
            for (int s = 0; s < ns; ++s)
                Cz_[(nz-1)*ns+s] = 0.0;
        } else {
            for (int s = 0; s < ns; ++s) {
                Bz_[(nz-1)*ns+s] -= Cz_[(nz-1)*ns+s];
                Cz_[(nz-1)*ns+s]  = 0.0;
            }
        }
    }

    {
        double t0 = MPI_Wtime();
        fdma_z_->set_rho(Az_.data(), Bz_.data(), Cz_.data(), ns);
        fdma_z_->solve(Az_.data(), Bz_.data(), Cz_.data(), Dz_.data());
        if (step_count_ > 20000) tdma_time_ += MPI_Wtime() - t0;
    }

    for (int j = 1; j <= ny; ++j)
        for (int i = 1; i <= nx; ++i) {
            const int s = (i-1)*ny + (j-1);
            for (int k = 1; k <= nz; ++k)
                dQ(i,j,k) = Dz_[(k-1)*ns + s];
        }
}

// ---------------------------------------------------------------------------
// cross_BW_V_  (MPM-STD blockLdV_kernel — core_momentum.f90:1624-1700)
//
//   dV -= dt · M23 · dW   where  M23·dW = convection_cross + stress_tensor_cross
//
//   M23dW = 0.25·(dwm5·∂V/∂z|_5 + dwm6·∂V/∂z|_6)
//         - nu_h·(∂(dW)/∂y|_6 - ∂(dW)/∂y|_5) / dz_k
//
//   dwm5/dwm6 : dW interpolated to V's location at z-face k-1/2 / k+1/2.
//   ∂(dW)/∂y at z-face uses (dW(j) - dW(jm))/dmy[j] — both at z-face k or kp.
//
//   Wall flags (kvm/kvp) zero out cross terms touching wall ghost cells.
// ---------------------------------------------------------------------------
void MomentumSolver::cross_BW_V_(Field<double>& dV, const Field<double>& V,
                                 const Field<double>& dW, double dt)
{
    const int nx = nx_, ny = ny_, nz = nz_;
    const auto& dy  = grid_->dx (1);   const auto& dmy = grid_->dmx(1);
    const auto& dz  = grid_->dx (2);   const auto& dmz = grid_->dmx(2);
    const double nu_h = 0.5 * inv_Re_;

    for (int k = 1; k <= nz; ++k) {
        const double kvm = (k == 1  && rank_z_ == 0)        ? 0.0 : 1.0;
        const double kvp = (k == nz && rank_z_ == np3_ - 1) ? 0.0 : 1.0;
        const int km = k - 1, kp = k + 1;

        for (int j = 1; j <= ny; ++j) {
            const int jm = j - 1;

            for (int i = 1; i <= nx; ++i) {
                // dW interpolated to V location (V at y-face j-1/2): weight in y by dy
                const double dwm5 = (dy[jm]*dW(i,j,k)  + dy[j]*dW(i,jm,k))  / dmy[j] * 0.5;
                const double dwm6 = (dy[jm]*dW(i,j,kp) + dy[j]*dW(i,jm,kp)) / dmy[j] * 0.5;

                // ∂V/∂z at z-faces k-1/2 and k+1/2
                const double dvdz5 = (V(i,j,k)  - V(i,j,km)) / dmz[k];
                const double dvdz6 = (V(i,j,kp) - V(i,j,k))  / dmz[kp];

                // ∂(dW)/∂y at z-faces (dW is on z-face, differentiate in y between j-1, j)
                const double ddwdy5 = (dW(i,j,k)  - dW(i,jm,k))  / dmy[j];
                const double ddwdy6 = (dW(i,j,kp) - dW(i,jm,kp)) / dmy[j];

                const double M23dW = 0.25*(dwm5*kvm*dvdz5 + dwm6*kvp*dvdz6)
                                   - nu_h*(ddwdy6*kvp - ddwdy5*kvm) / dz[k];

                dV(i,j,k) -= dt * M23dW;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// cross_BW_U_  (MPM-STD blockLdU_kernel — core_momentum.f90:1702-1797)
//
//   dU -= dt · (M12 · dV + M13 · dW)
//
//   M12dV = 0.25·(dvm3·∂U/∂y|_3 + dvm4·∂U/∂y|_4)
//         - nu_h·(∂(dV)/∂x|_4 - ∂(dV)/∂x|_3) / dy_j
//   M13dW = 0.25·(dwm5·∂U/∂z|_5 + dwm6·∂U/∂z|_6)
//         - nu_h·(∂(dW)/∂x|_6 - ∂(dW)/∂x|_5) / dz_k
//
//   dvm3/dvm4 : dV interpolated to U's location at y-faces j-1/2 / j+1/2
//   dwm5/dwm6 : dW interpolated to U's location at z-faces k-1/2 / k+1/2
//
//   Periodic in y → jum=jup=1 always.  Wall flags (kum/kup) for z only.
// ---------------------------------------------------------------------------
void MomentumSolver::cross_BW_U_(Field<double>& dU, const Field<double>& U,
                                 const Field<double>& dV, const Field<double>& dW, double dt)
{
    const int nx = nx_, ny = ny_, nz = nz_;
    const auto& dx  = grid_->dx (0);   const auto& dmx = grid_->dmx(0);
    const auto& dy  = grid_->dx (1);   const auto& dmy = grid_->dmx(1);
    const auto& dz  = grid_->dx (2);   const auto& dmz = grid_->dmx(2);
    const double nu_h = 0.5 * inv_Re_;

    for (int k = 1; k <= nz; ++k) {
        const double kum = (k == 1  && rank_z_ == 0)        ? 0.0 : 1.0;
        const double kup = (k == nz && rank_z_ == np3_ - 1) ? 0.0 : 1.0;
        const int km = k - 1, kp = k + 1;

        for (int j = 1; j <= ny; ++j) {
            const double jum = 1.0;   // periodic y → no wall
            const double jup = 1.0;
            const int jm = j - 1, jp = j + 1;

            for (int i = 1; i <= nx; ++i) {
                const int im = i - 1;

                // dV interpolated to U location (U at x-face i-1/2): weight in x by dx
                const double dvm3 = (dx[im]*dV(i,j, k) + dx[i]*dV(im,j, k)) / dmx[i] * 0.5;
                const double dvm4 = (dx[im]*dV(i,jp,k) + dx[i]*dV(im,jp,k)) / dmx[i] * 0.5;

                // ∂U/∂y at y-faces (U at cell-center in y, but x-face in x)
                const double dudy3 = (U(i,j, k) - U(i,jm,k)) / dmy[j];
                const double dudy4 = (U(i,jp,k) - U(i,j, k)) / dmy[jp];

                // ∂(dV)/∂x at y-faces (dV is on y-face, differentiate in x between i-1, i)
                const double dvmdx3 = (dV(i,j, k) - dV(im,j, k)) / dmx[i];
                const double dvmdx4 = (dV(i,jp,k) - dV(im,jp,k)) / dmx[i];

                // dW interpolated to U location: weight in x by dx
                const double dwm5 = (dx[im]*dW(i,j,k)  + dx[i]*dW(im,j,k))  / dmx[i] * 0.5;
                const double dwm6 = (dx[im]*dW(i,j,kp) + dx[i]*dW(im,j,kp)) / dmx[i] * 0.5;

                // ∂U/∂z at z-faces
                const double dudz5 = (U(i,j,k)  - U(i,j,km)) / dmz[k];
                const double dudz6 = (U(i,j,kp) - U(i,j,k))  / dmz[kp];

                // ∂(dW)/∂x at z-faces
                const double dwmdx5 = (dW(i,j,k)  - dW(im,j,k))  / dmx[i];
                const double dwmdx6 = (dW(i,j,kp) - dW(im,j,kp)) / dmx[i];

                const double M12dV = 0.25*(dvm3*jum*dudy3 + dvm4*jup*dudy4)
                                   - nu_h*(dvmdx4*jup - dvmdx3*jum) / dy[j];
                const double M13dW = 0.25*(dwm5*kum*dudz5 + dwm6*kup*dudz6)
                                   - nu_h*(dwmdx6*kup - dwmdx5*kum) / dz[k];

                dU(i,j,k) -= dt * (M12dV + M13dW);
            }
        }
    }
}

// ---------------------------------------------------------------------------
void MomentumSolver::advance(Field<double>& U, Field<double>& V, Field<double>& W,
                             const Field<double>& P,
                             double dt, double mean_dPdx)
{
    const int nx = nx_, ny = ny_, nz = nz_;

    ++step_count_;

    const double t_adv0 = MPI_Wtime();

    fdma_x_->set_eps_constant(dt);
    fdma_y_->set_eps_constant(dt);
    fdma_z_->set_eps_constant(dt);

    // MPM-STD core_momentum + blockLdU pattern (submodule.f90:195-232):
    //   1) Independently solve each component's ADI (Jacobi base)
    //   2) Halo-exchange dU, dV, dW so cross-coupling reads valid neighbors
    //   3) Cross-couple V eq with dW          → halo-exchange dV
    //   4) Cross-couple U eq with dV (fresh) and dW
    //   5) Apply all increments atomically
    // W eq has no cross-BW (chain head, like MPM-STD which omits blockLdW).

    // --- (1) Independent ADI for each component --------------------
    compute_rhs_(COMP_U, dU_, U, V, W, P, dt, mean_dPdx);
    adi_sweep_x_(COMP_U, dU_, dt, U, V, W);
    adi_sweep_y_(COMP_U, dU_, dt, U, V, W);
    adi_sweep_z_(COMP_U, dU_, dt, U, V, W);

    compute_rhs_(COMP_V, dV_, U, V, W, P, dt, mean_dPdx);
    adi_sweep_x_(COMP_V, dV_, dt, U, V, W);
    adi_sweep_y_(COMP_V, dV_, dt, U, V, W);
    adi_sweep_z_(COMP_V, dV_, dt, U, V, W);

    compute_rhs_(COMP_W, dW_, U, V, W, P, dt, mean_dPdx);
    adi_sweep_x_(COMP_W, dW_, dt, U, V, W);
    adi_sweep_y_(COMP_W, dW_, dt, U, V, W);
    adi_sweep_z_(COMP_W, dW_, dt, U, V, W);

    // --- (2) Refresh halos of all increment fields -----------------
    // (MPM-STD: cuda_subdomain_ghostcell_update at end of cuda_momentum_solvedU)
    halo_->exchange(dU_);
    halo_->exchange(dV_);
    halo_->exchange(dW_);

    // --- (3) Cross-couple V using dW; refresh dV halo --------------
    cross_BW_V_(dV_, V, dW_, dt);
    halo_->exchange(dV_);

    // --- (4) Cross-couple U using dV (fresh) and dW ----------------
    cross_BW_U_(dU_, U, dV_, dW_, dt);

    // --- (5) Apply all increments atomically -----------------------
    for (int k = 1; k <= nz; ++k)
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i) {
                U(i,j,k) += dU_(i,j,k);
                V(i,j,k) += dV_(i,j,k);
                W(i,j,k) += dW_(i,j,k);
            }

    if (step_count_ > 20000) momentum_time_ += MPI_Wtime() - t_adv0;
}

} // namespace channel
