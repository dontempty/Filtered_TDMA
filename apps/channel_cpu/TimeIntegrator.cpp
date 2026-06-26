// channel/TimeIntegrator.cpp — Top-level time loop.
//
// Output timing (v2 convention):
//   monitor:     step % nmonitor == 0
//   stats accum: step >= nstat_start && (step-nstat_start) % nstat      == 0
//   stats file:  out_stats  && step >= nstat_start
//                          && (step-nstat_start) % nout_stats == 0
//   field file:  out_field  && step % nout == 0
//   restart:     ContinueFileout && (step-nstat_start) % nout_stats == 0

#include "TimeIntegrator.hpp"

#include "BoundaryCondition.hpp"
#include "ChannelForcing.hpp"
#include "Config.hpp"
#include "FieldOutput.hpp"
#include "Grid.hpp"
#include "HaloExchanger.hpp"
#include "MomentumSolver.hpp"
#include "MpiTopology.hpp"
#include "PressureSolver.hpp"
#include "RestartIO.hpp"
#include "Statistics.hpp"
#include "Subdomain.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <mpi.h>

namespace channel {

TimeIntegrator::TimeIntegrator(const Config&       cfg,
                               const MpiTopology&  topo,
                               const Subdomain&    sub,
                               const Grid&         grid,
                               const HaloExchanger& halo,
                               const BoundaryCondition& bc,
                               MomentumSolver&     momentum,
                               PressureSolver&     pressure,
                               ChannelForcing&     forcing,
                               Statistics&         stats,
                               RestartIO&          restart,
                               FieldOutput&        field_out)
    : cfg_(cfg), topo_(topo), sub_(sub), grid_(grid),
      halo_(halo), bc_(bc),
      momentum_(momentum), pressure_(pressure), forcing_(forcing),
      stats_(stats), restart_(restart), field_out_(field_out) {}

// ---- CFL time step -------------------------------------------------------
double TimeIntegrator::cfl_dt_(const Field<double>& U, const Field<double>& V,
                               const Field<double>& W) const
{
    const int nx = sub_.nx(), ny = sub_.ny(), nz = sub_.nz();
    const auto& dx = grid_.dx(0);
    const auto& dy = grid_.dx(1);
    const auto& dz = grid_.dx(2);

    double local_max = 0.0;
    for (int k = 1; k <= nz; ++k)
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i) {
                double r = std::fabs(U(i, j, k)) / dx[i]
                         + std::fabs(V(i, j, k)) / dy[j]
                         + std::fabs(W(i, j, k)) / dz[k];
                if (r > local_max) local_max = r;
            }
    double global_max = 0.0;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, topo_.cart());
    if (global_max < 1.0e-30) return cfg_.dtStart;
    return cfg_.MaxCFL / global_max;
}

// ---- spectral radius ρ ---------------------------------------------------
// Gershgorin bound for the z-direction BW TDMA (pure diffusion, no convection):
//   αm = ν_h·dt / (dz[k]·dmz[k]),  αp = ν_h·dt / (dz[k]·dmz[k+1])
//   ρ_k = max(αm, αp) / (αm + αp + 1)
// Matches FilteredTDMA cal_J_v2: max of |am/ac|, |ap/ac| over skip=2 interior rows.
// Wall-adjacent cells (global k ≤ 2 or k ≥ n3m-1) are excluded, same as FilteredTDMA.
// ρ_max: from cells closest to walls (among non-skipped rows, smallest dz·dmz)
// ρ_min: from centre cells (largest dz·dmz)
std::pair<double,double> TimeIntegrator::rho_diagnostic_(double dt) const
{
    const auto& dz  = grid_.dx(2);
    const auto& dmz = grid_.dmx(2);
    const double nu_h  = 0.5 / cfg_.Re_b;
    const int nz       = sub_.nz();
    const int ista_z   = sub_.ista(2);    // global 1-based start index for this rank
    const int n3m      = sub_.global_n(2);
    const int skip     = 2;

    double rho_max_loc = 0.0, rho_min_loc = 1.0;
    for (int k = 1; k <= nz; ++k) {
        int gk = ista_z + k - 1;                          // global z index (1-based)
        if (gk < skip + 1 || gk > n3m - skip) continue;  // exclude skip=2 wall rows

        double am = nu_h * dt / (dz[k] * dmz[k]);
        double ap = nu_h * dt / (dz[k] * dmz[k+1]);
        double rk = std::max(am, ap) / (am + ap + 1.0);
        if (rk > rho_max_loc) rho_max_loc = rk;
        if (rk < rho_min_loc) rho_min_loc = rk;
    }

    double rho_max = 0.0, rho_min = 1.0;
    MPI_Allreduce(&rho_max_loc, &rho_max, 1, MPI_DOUBLE, MPI_MAX, topo_.cart());
    MPI_Allreduce(&rho_min_loc, &rho_min, 1, MPI_DOUBLE, MPI_MIN, topo_.cart());
    return { rho_max, rho_min };
}

// ---- wall shear stress ---------------------------------------------------
// Average |dU/dz| over both walls × ν.
double TimeIntegrator::wss_diagnostic_(const Field<double>& U) const
{
    const int nx = sub_.nx(), ny = sub_.ny(), nz = sub_.nz();
    const auto& dz = grid_.dx(2);
    const double inv_Re = 1.0 / cfg_.Re_b;

    bool is_bottom = (sub_.ista(2) == 1);
    bool is_top    = (sub_.ista(2) + nz - 1 == sub_.global_n(2));

    double loc = 0.0;
    if (is_bottom)
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i)
                loc += (U(i, j, 1) - 0.0) / (dz[1] * 0.5);
    if (is_top)
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i)
                loc += (U(i, j, nz) - 0.0) / (dz[nz] * 0.5);

    double global = 0.0;
    MPI_Allreduce(&loc, &global, 1, MPI_DOUBLE, MPI_SUM, topo_.cart());
    const int nxy = sub_.global_n(0) * sub_.global_n(1);
    return std::fabs(inv_Re * global / (2.0 * nxy));
}

// ---- bulk velocity -------------------------------------------------------
double TimeIntegrator::bulk_velocity_(const Field<double>& U) const
{
    const int nx = sub_.nx(), ny = sub_.ny(), nz = sub_.nz();
    const auto& dx = grid_.dx(0);
    const auto& dy = grid_.dx(1);
    const auto& dz = grid_.dx(2);

    double vol_local = 0.0, sum_local = 0.0;
    for (int k = 1; k <= nz; ++k)
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i) {
                double dV = dx[i] * dy[j] * dz[k];
                sum_local += 0.5 * (U(i, j, k) + U(i+1, j, k)) * dV;
                vol_local += dV;
            }
    double sums[2] = {sum_local, vol_local};
    double glob[2] = {0.0, 0.0};
    MPI_Allreduce(sums, glob, 2, MPI_DOUBLE, MPI_SUM, topo_.cart());
    return (glob[1] > 0.0) ? glob[0] / glob[1] : 0.0;
}

// ---- maximum divergence --------------------------------------------------
double TimeIntegrator::max_div_u_(const Field<double>& U,
                                   const Field<double>& V,
                                   const Field<double>& W) const
{
    const int nx = sub_.nx(), ny = sub_.ny(), nz = sub_.nz();
    const auto& dx = grid_.dx(0);
    const auto& dy = grid_.dx(1);
    const auto& dz = grid_.dx(2);

    double local_max = 0.0;
    for (int k = 1; k <= nz; ++k)
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i) {
                double d = std::fabs(
                    (U(i+1,j,k)-U(i,j,k))/dx[i] +
                    (V(i,j+1,k)-V(i,j,k))/dy[j] +
                    (W(i,j,k+1)-W(i,j,k))/dz[k]);
                if (d > local_max) local_max = d;
            }
    double global_max = 0.0;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, topo_.cart());
    return global_max;
}

// ---- debug: locate global max |U|, |V|, |W| each step ------------------
// Writes one row per call:
//   step, t, dt, max|U|, U_i, U_j, U_k, max|V|, V_i, V_j, V_k,
//                       max|W|, W_i, W_j, W_k,  max_div, div_i, div_j, div_k
// (i,j,k are GLOBAL 1-based indices.  Owner rank is gathered via MAXLOC.)
void TimeIntegrator::write_max_velocity_debug_(const Field<double>& U,
                                               const Field<double>& V,
                                               const Field<double>& W,
                                               long step, double dt, double t,
                                               std::FILE* fp,
                                               bool verbose_stdout) const
{
    const int nx = sub_.nx(), ny = sub_.ny(), nz = sub_.nz();
    const auto& dx = grid_.dx(0);
    const auto& dy = grid_.dx(1);
    const auto& dz = grid_.dx(2);
    const int ix = sub_.ista(0) - 1;
    const int iy = sub_.ista(1) - 1;
    const int iz = sub_.ista(2) - 1;

    // Locate local max of |U|, |V|, |W|, |divU|
    double mu = 0, mv = 0, mw = 0, md = 0;
    int iu=0,ju=0,ku=0, iv=0,jv=0,kv=0, iw=0,jw=0,kw=0, id=0,jd=0,kd=0;
    for (int k = 1; k <= nz; ++k)
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i) {
                double au = std::fabs(U(i,j,k));
                double av = std::fabs(V(i,j,k));
                double aw = std::fabs(W(i,j,k));
                double ad = std::fabs((U(i+1,j,k)-U(i,j,k))/dx[i]
                                    + (V(i,j+1,k)-V(i,j,k))/dy[j]
                                    + (W(i,j,k+1)-W(i,j,k))/dz[k]);
                if (au > mu) { mu = au; iu=i; ju=j; ku=k; }
                if (av > mv) { mv = av; iv=i; jv=j; kv=k; }
                if (aw > mw) { mw = aw; iw=i; jw=j; kw=k; }
                if (ad > md) { md = ad; id=i; jd=j; kd=k; }
            }

    // MPI_MAXLOC for each field — gives global max, owner rank, and global (i,j,k).
    auto reduce_one = [&](double local_val, int li, int lj, int lk,
                          double& g_val, int& gi, int& gj, int& gk, int& g_rank) {
        struct { double v; int r; } in, out;
        int my_rank = topo_.rank();
        in.v = local_val; in.r = my_rank;
        MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, topo_.cart());
        g_val = out.v;
        g_rank = out.r;
        int loc[3] = { li + ix, lj + iy, lk + iz };
        MPI_Bcast(loc, 3, MPI_INT, out.r, topo_.cart());
        gi = loc[0]; gj = loc[1]; gk = loc[2];
    };

    double gu=0, gv=0, gw=0, gd=0;
    int Giu=0,Gju=0,Gku=0, Giv=0,Gjv=0,Gkv=0, Giw=0,Gjw=0,Gkw=0, Gid=0,Gjd=0,Gkd=0;
    int Ru=0, Rv=0, Rw=0, Rd=0;
    reduce_one(mu, iu, ju, ku, gu, Giu, Gju, Gku, Ru);
    reduce_one(mv, iv, jv, kv, gv, Giv, Gjv, Gkv, Rv);
    reduce_one(mw, iw, jw, kw, gw, Giw, Gjw, Gkw, Rw);
    reduce_one(md, id, jd, kd, gd, Gid, Gjd, Gkd, Rd);

    if (topo_.rank() == 0) {
        if (fp) {
            std::fprintf(fp,
                "%ld,%.6e,%.6e,"
                "%.9e,%d,%d,%d,"
                "%.9e,%d,%d,%d,"
                "%.9e,%d,%d,%d,"
                "%.9e,%d,%d,%d\n",
                step, t, dt,
                gu, Giu, Gju, Gku,
                gv, Giv, Gjv, Gkv,
                gw, Giw, Gjw, Gkw,
                gd, Gid, Gjd, Gkd);
            std::fflush(fp);
        }
        if (verbose_stdout) {
            const int n3m = sub_.global_n(2);
            std::printf("\n========================================================\n");
            std::printf("[DIAGNOSTIC] step=%ld  t=%.6e  dt=%.6e\n", step, t, dt);
            std::printf("--------------------------------------------------------\n");
            std::printf("  maxDivU = %.6e   @ global (i,j,k) = (%d,%d,%d)   rank=%d\n",
                        gd, Gid, Gjd, Gkd, Rd);
            std::printf("  max|U|  = %.6e   @ global (i,j,k) = (%d,%d,%d)   rank=%d  k_wall=%d\n",
                        gu, Giu, Gju, Gku, Ru, std::min(Gku, n3m - Gku + 1));
            std::printf("  max|V|  = %.6e   @ global (i,j,k) = (%d,%d,%d)   rank=%d  k_wall=%d\n",
                        gv, Giv, Gjv, Gkv, Rv, std::min(Gkv, n3m - Gkv + 1));
            std::printf("  max|W|  = %.6e   @ global (i,j,k) = (%d,%d,%d)   rank=%d  k_wall=%d\n",
                        gw, Giw, Gjw, Gkw, Rw, std::min(Gkw, n3m - Gkw + 1));
            std::printf("  (k_wall = distance from nearest wall in cell counts; 1 = first cell at wall)\n");
            std::printf("========================================================\n");
            std::fflush(stdout);
        }
    }
}

// ---- main time loop ------------------------------------------------------
void TimeIntegrator::run(Field<double>& U, Field<double>& V, Field<double>& W,
                         Field<double>& P, RestartState& state)
{
    const bool root = (topo_.rank() == 0);
    double t  = state.time;
    double dt = (state.dt > 0) ? state.dt : cfg_.dtStart;
    long  step = state.step;
    const long step_end = step + cfg_.Timestepmax;
    const int  mstr = (cfg_.nmonitor > 0) ? cfg_.nmonitor : 1;

    // Open monitor file (append)
    std::FILE* mon_fp  = nullptr;
    std::FILE* wss_fp  = nullptr;
    if (root) {
        std::string mon_path = cfg_.dir_instantfield + "Monitor_Channel.plt";
        mon_fp = std::fopen(mon_path.c_str(), "a");
        if (mon_fp && std::ftell(mon_fp) == 0)
            std::fprintf(mon_fp,
                "VARIABLES=\"Timestep\" \"Time\" \"dt\""
                " \"CFL\" \"maxDivU\" \"WSS\" \"u_tau\" \"U_b\""
                " \"rho_max\" \"rho_min\"\n");

        // WSS history (like reference stats/wss_history.dat)
        std::string wss_path = cfg_.dir_statistics + "/wss_history.dat";
        wss_fp = std::fopen(wss_path.c_str(), "a");
        if (wss_fp && std::ftell(wss_fp) == 0)
            std::fprintf(wss_fp,
                "# %10s %12s %12s %12s %12s %12s %12s %12s %12s\n",
                "step", "time", "dt", "wss", "u_tau", "div_max", "U_b",
                "rho_max", "rho_min");
    }

    // (Debug CSV is no longer opened up-front.  It is created on demand only
    //  when an abort fires — see emergency-dump block below.)

    // Pre-loop: ensure valid ghosts and BCs
    halo_.exchange(U); halo_.exchange(V); halo_.exchange(W);
    halo_.exchange(P);
    bc_.apply(U, V, W);

    while (step < step_end) {

        double dt_cfl    = cfl_dt_(U, V, W);
        dt = std::min(dt_cfl, cfg_.dtStart);
        double mean_dPdx = forcing_.mean_dPdx();

        // [C] Momentum
        momentum_.advance(U, V, W, P, dt, mean_dPdx);

        // [D] Mass-flow forcing — uniform shift on U interior (no halo access)
        mean_dPdx = forcing_.correct(U, dt);

        // [E] Partial ghost refresh for pressure solve (face halos only)
        halo_.exchange_axis(U, 0);
        halo_.exchange_axis(V, 1);
        halo_.exchange_axis(W, 2);
        bc_.apply(U, V, W);

        // [F] Pressure solve
        pressure_.solve(U, V, W, P, dt);

        // [H+I+J] Full ghost restore (STATE INVARIANT for next iter)
        halo_.exchange(U); halo_.exchange(V); halo_.exchange(W);
        halo_.exchange(P);
        bc_.apply(U, V, W);

        // ---- Divergence safety: emergency abort if |divU| > 1e-10 ----
        const double divU_global = max_div_u_(U, V, W);
        const bool abort_now = (divU_global > 1.0e-10);

        // Statistics accumulation — matches v2: every nstat steps from nstat_start
        if (step >= cfg_.nstat_start &&
            ((step - cfg_.nstat_start) % cfg_.nstat == 0))
            stats_.accumulate(U, V, W, P);

        ++step;
        t += dt;

        // ---- Monitor ----
        if (step % mstr == 0) {
            auto [rho_max, rho_min] = rho_diagnostic_(dt);
            double wss     = wss_diagnostic_(U);
            double u_tau   = std::sqrt(wss);
            double U_b     = bulk_velocity_(U);
            double max_div = max_div_u_(U, V, W);
            if (root) {
                if (((step / mstr) % 10) == 1) {
                    std::printf("\n");
                    std::printf("%12s %12s %12s %12s %12s %12s %12s %12s %12s\n",
                                "Timestep","Time","dt",
                                "maxDivU","WSS","u_tau","U_b","rho_max","rho_min");
                }
                std::printf(
                    "%12ld %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e\n",
                    step, t, dt,
                    max_div, wss, u_tau, U_b, rho_max, rho_min);
                std::fflush(stdout);

                if (mon_fp) {
                    std::fprintf(mon_fp,
                        "%d %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n",
                        (int)step, t, dt,
                        max_div, wss, u_tau, U_b, rho_max, rho_min);
                    std::fflush(mon_fp);
                }
                if (wss_fp) {
                    std::fprintf(wss_fp,
                        "  %10d %12.6e %12.4e %12.6e %12.6e %12.4e %12.6e %12.6e %12.6e\n",
                        (int)step, t, dt, wss, u_tau, max_div, U_b, rho_max, rho_min);
                    std::fflush(wss_fp);
                }
            }
        }

        // ---- Statistics write (numbered files) — matches v2 ----
        if (cfg_.out_stats && step >= cfg_.nstat_start
                && ((step - cfg_.nstat_start) % cfg_.nout_stats == 0)
                && stats_.samples() > 0) {
            char sname[512];
            std::snprintf(sname, sizeof(sname), "%s/stats_%08ld.dat",
                          cfg_.dir_statistics.c_str(), step);
            stats_.write(sname, (int)step, cfg_.Re_b, false);
        }

        // ---- 3-D field output — same start/interval pattern as stats ----
        if (cfg_.out_field && cfg_.nout > 0
                && step >= cfg_.nfield_start
                && ((step - cfg_.nfield_start) % cfg_.nout == 0))
            field_out_.write_field_3d(U, V, W, P, (int)step);

        // ---- Restart (piggy-backs on stats write cadence) ----
        if (cfg_.ContinueFileout && step >= cfg_.nstat_start
                && ((step - cfg_.nstat_start) % cfg_.nout_stats == 0)) {
            state.time = t; state.dt = dt; state.step = step; state.dPdx = mean_dPdx;
            restart_.write(cfg_.dir_cont_fileout, U, V, W, P, state, step);
        }

        // ---- Emergency abort: divU exploded → dump everything we can ----
        if (abort_now) {
            if (root) {
                std::printf("\n[ABORT] step=%ld  maxDivU=%.6e > 1.0e-10 "
                            "— emergency dump and exit\n",
                            step, divU_global);
                std::fflush(stdout);
            }
            // Open (create) the debug CSV only on abort; write one row + header
            // and emit a verbose location block to stdout in the same call.
            std::FILE* abort_fp = nullptr;
            if (root) {
                std::string p = cfg_.dir_statistics + "/max_velocity_debug.csv";
                abort_fp = std::fopen(p.c_str(), "a");
                if (abort_fp && std::ftell(abort_fp) == 0) {
                    std::fprintf(abort_fp,
                        "step,t,dt,"
                        "maxU,iU,jU,kU,"
                        "maxV,iV,jV,kV,"
                        "maxW,iW,jW,kW,"
                        "maxDivU,iD,jD,kD\n");
                }
            }
            write_max_velocity_debug_(U, V, W, step, dt, t, abort_fp, true);
            if (root && abort_fp) std::fclose(abort_fp);
            // stats: write whatever has been accumulated so far
            if (stats_.samples() > 0) {
                char sname[512];
                std::snprintf(sname, sizeof(sname), "%s/stats_abort_%08ld.dat",
                              cfg_.dir_statistics.c_str(), step);
                stats_.write(sname, (int)step, cfg_.Re_b, false);
            }
            // 3-D field snapshot at the abort step (for post-mortem inspection)
            field_out_.write_field_3d(U, V, W, P, (int)step);
            // restart write — regardless of ContinueFileout setting
            state.time = t; state.dt = dt; state.step = step;
            state.dPdx = mean_dPdx;
            restart_.write(cfg_.dir_cont_fileout, U, V, W, P, state, step);
            break;
        }
    }

    state.time = t; state.dt = dt; state.step = step;
    state.dPdx = forcing_.mean_dPdx();

    // Final statistics
    if (stats_.samples() > 0) {
        char sname[512];
        std::snprintf(sname, sizeof(sname), "%s/stats_final_%08ld.dat",
                      cfg_.dir_statistics.c_str(), step);
        stats_.write(sname, (int)step, cfg_.Re_b, false);
    }

    if (cfg_.ContinueFileout)
        restart_.write(cfg_.dir_cont_fileout, U, V, W, P, state, state.step);

    // ---- Momentum / TDMA timing report (accumulated from step 20001 onward) ----
    {
        double loc[5] = { momentum_.momentum_time(),
                          momentum_.tdma_x_time(),
                          momentum_.tdma_y_time(),
                          momentum_.tdma_z_time(),
                          momentum_.tdma_time() };
        double glo[5] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
        MPI_Reduce(loc, glo, 5, MPI_DOUBLE, MPI_MAX, 0, topo_.cart());
        if (root) {
            const double mom = glo[0];
            const double tx  = glo[1], ty = glo[2], tz = glo[3], ttot = glo[4];
            const double frac = (mom > 0.0) ? 100.0 * ttot / mom : 0.0;
            std::printf("[Momentum time (step>20000)] %.6e s  (max over ranks)\n", mom);
            std::printf("[TDMA  x time (step>20000)]  %.6e s  (max over ranks)\n", tx);
            std::printf("[TDMA  y time (step>20000)]  %.6e s  (max over ranks)\n", ty);
            std::printf("[TDMA  z time (step>20000)]  %.6e s  (max over ranks)\n", tz);
            std::printf("[TDMA total time (step>20000)] %.6e s  (max over ranks, %.2f%% of momentum)\n",
                        ttot, frac);
        }
    }

    // ---- Per-step timing CSV: one file per rank ----
    {
        int my_rank = 0;
        MPI_Comm_rank(topo_.cart(), &my_rank);
        char path[512];
        std::snprintf(path, sizeof(path), "%s/tdma_timing_rank%04d.csv",
                      cfg_.dir_statistics.c_str(), my_rank);
        momentum_.write_timing_csv(path);
    }

    if (root) {
        if (mon_fp) std::fclose(mon_fp);
        if (wss_fp) std::fclose(wss_fp);
    }
}

} // namespace channel
