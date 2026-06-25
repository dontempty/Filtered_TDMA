// channel/main.cpp — Channel flow solver entry point.

#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>

#include "BoundaryCondition.hpp"
#include "ChannelForcing.hpp"
#include "Config.hpp"
#include "Field.hpp"
#include "FieldOutput.hpp"
#include "Grid.hpp"
#include "HaloExchanger.hpp"
#include "MomentumSolver.hpp"
#include "MpiTopology.hpp"
#include "PressureSolver.hpp"
#include "RestartIO.hpp"
#include "Statistics.hpp"
#include "Subdomain.hpp"
#include "TimeIntegrator.hpp"

namespace {

// MPM-STD cuda_momentum_init_channel pattern (core_momentum.f90:180-383):
//   1. Random uniform[-0.5,0.5] perturbation on ALL THREE velocity components
//   2. Subtract spatial mean from each perturbation field (mean-zero)
//   3. Add parabolic Poiseuille to U; perturbation only to V, W
//   4. Re-normalize bulk: U -= (Ub_total - Ub),  V -= Vb_total,  W -= Wb_total
// Sub-critical Re_b (< 5772 plane channel) requires 3-component bypass-transition
// noise; without V,W perturbation the flow stays linearly stable forever.
void laminar_init(channel::Field<double>& U, channel::Field<double>& V,
                  channel::Field<double>& W, channel::Field<double>& P,
                  const channel::Subdomain& sub, const channel::Grid& grid,
                  double Ly, double Lz, double Ub, double pert, int init_mode,
                  MPI_Comm comm)
{
    const int nx = sub.nx(), ny = sub.ny(), nz = sub.nz();
    const auto& yc = grid.x(1);
    const auto& zc = grid.x(2);
    const auto& dx = grid.dx(0);
    const auto& dy = grid.dx(1);
    const auto& dz = grid.dx(2);
    const double Umax = 1.5 * Ub;
    const double half = 0.5 * Lz;

    int my_rank = 0;
    MPI_Comm_rank(comm, &my_rank);
    // Non-deterministic seed (matches MPM-STD's default Fortran random_seed()).
    // Mixing random_device entropy with the rank ensures both per-run and
    // per-rank distinguishability.
    std::random_device rd;
    const std::uint64_t seed = static_cast<std::uint64_t>(rd())
                              ^ (static_cast<std::uint64_t>(my_rank) * 0x9E3779B97F4A7C15ULL);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> rnd(-0.5, 0.5);   // matches MPM-STD: uniform[0,1]-0.5
    P.fill(0.0);

    // Volume-weighted global mean of a field (mean-zero + bulk renorm).
    auto bulk = [&](const channel::Field<double>& F) {
        double s = 0.0, v = 0.0;
        for (int k = 1; k <= nz; ++k)
            for (int j = 1; j <= ny; ++j)
                for (int i = 1; i <= nx; ++i) {
                    double dV = dx[i] * dy[j] * dz[k];
                    s += F(i, j, k) * dV;
                    v += dV;
                }
        double pkt[2] = {s, v}, tot[2] = {0.0, 0.0};
        MPI_Allreduce(pkt, tot, 2, MPI_DOUBLE, MPI_SUM, comm);
        return (tot[1] > 0.0) ? tot[0] / tot[1] : 0.0;
    };

    if (init_mode == 1) {
        // ----- sin/cos streamwise vortices (divergence-free) + small 3D noise -----
        // psi(y,z) = A cos(beta*y) sin^2(pi*z/Lz):
        //   V =  d psi/dz = A cos(beta*y)(pi/Lz) sin(2 pi z/Lz)
        //   W = -d psi/dy = A beta sin(beta*y) sin^2(pi z/Lz)
        // => divergence-free in (y,z); V=W=0 at z=0,Lz (no-slip). Counter-rotating
        //    rolls pump wall-normal momentum -> streaks by lift-up = far more
        //    effective bypass-transition trigger than broadband white noise.
        const double nbeta = std::max(1.0, std::round(Ly / Lz));  // ~channel-height rolls
        const double beta  = 2.0 * M_PI * nbeta / Ly;
        const double kz    = M_PI / Lz;
        double locmax = 0.0;
        for (int k = 1; k <= nz; ++k)
            for (int j = 1; j <= ny; ++j)
                for (int i = 1; i <= nx; ++i) {
                    double s1 = std::sin(kz * zc[k]);
                    double sV = std::cos(beta * yc[j]) * kz * std::sin(2.0 * kz * zc[k]);
                    double sW = beta * std::sin(beta * yc[j]) * s1 * s1;
                    V(i, j, k) = sV;
                    W(i, j, k) = sW;
                    locmax = std::max(locmax, std::max(std::fabs(sV), std::fabs(sW)));
                }
        double gmax = locmax;
        MPI_Allreduce(MPI_IN_PLACE, &gmax, 1, MPI_DOUBLE, MPI_MAX, comm);
        const double scale = (gmax > 0.0) ? (pert * Umax) / gmax : 0.0;
        const double noise = 0.1 * pert * Umax;
        for (int k = 1; k <= nz; ++k) {
            double zr = (zc[k] - half) / half;
            double Up = Umax * std::max(0.0, 1.0 - zr * zr);
            for (int j = 1; j <= ny; ++j)
                for (int i = 1; i <= nx; ++i) {
                    U(i, j, k) = Up                 + noise * rnd(rng);
                    V(i, j, k) = scale * V(i, j, k) + noise * rnd(rng);
                    W(i, j, k) = scale * W(i, j, k) + noise * rnd(rng);
                }
        }
        if (my_rank == 0)
            std::printf("[laminar_init] mode=VORTEX  nbeta=%.0f  roll_amp=%.3g*Umax (+0.1x noise)\n",
                        nbeta, pert);
    } else {
        // ----- random uniform[-0.5,0.5] noise on all 3 components (default) -----
        if (my_rank == 0)
            std::printf("[laminar_init] mode=RANDOM  seed(rank0=%llu)\n",
                        (unsigned long long)seed);
        for (int k = 1; k <= nz; ++k)
            for (int j = 1; j <= ny; ++j)
                for (int i = 1; i <= nx; ++i) {
                    U(i, j, k) = rnd(rng);
                    V(i, j, k) = rnd(rng);
                    W(i, j, k) = rnd(rng);
                }
        double Um0 = bulk(U), Vm0 = bulk(V), Wm0 = bulk(W);
        for (int k = 1; k <= nz; ++k)
            for (int j = 1; j <= ny; ++j)
                for (int i = 1; i <= nx; ++i) {
                    U(i, j, k) -= Um0; V(i, j, k) -= Vm0; W(i, j, k) -= Wm0;
                }
        for (int k = 1; k <= nz; ++k) {
            double zr = (zc[k] - half) / half;
            double Up = Umax * std::max(0.0, 1.0 - zr * zr);
            for (int j = 1; j <= ny; ++j)
                for (int i = 1; i <= nx; ++i) {
                    U(i, j, k) = Up + pert * Umax * U(i, j, k);
                    V(i, j, k) =      pert * Umax * V(i, j, k);
                    W(i, j, k) =      pert * Umax * W(i, j, k);
                }
        }
    }

    // ----- re-normalize bulk velocity to exactly (Ub, 0, 0) (shared) -----
    double Um = bulk(U), Vm = bulk(V), Wm = bulk(W);
    const double dU = Ub - Um;
    for (int k = 1; k <= nz; ++k)
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i) {
                U(i, j, k) += dU;
                V(i, j, k) -= Vm;
                W(i, j, k) -= Wm;
            }
    if (my_rank == 0)
        std::printf("[laminar_init] pert=%.3g  bulk_pre=(%.3e,%.3e,%.3e) -> (%.3e,0,0)\n",
                    pert, Um, Vm, Wm, Ub);
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 2) {
        if (rank == 0)
            std::fprintf(stderr, "usage: %s <PARA_INPUT.dat>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rc = 0;
    {
        auto cfg = channel::Config::load(argv[1], MPI_COMM_WORLD);
        if (rank == 0) {
            cfg.print();
            namespace fs = std::filesystem;
            for (const auto& d : { cfg.dir_cont_filein, cfg.dir_cont_fileout,
                                   cfg.dir_instantfield, cfg.dir_statistics }) {
                std::error_code ec;
                fs::create_directories(d, ec);
                if (ec)
                    std::fprintf(stderr, "[main] warning: mkdir '%s': %s\n",
                                 d.c_str(), ec.message().c_str());
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        channel::MpiTopology topo(MPI_COMM_WORLD,
                                  cfg.np1, cfg.np2, cfg.np3,
                                  cfg.pbc1, cfg.pbc2, cfg.pbc3);
        if (rank == 0) topo.print();

        channel::Subdomain sub(cfg, topo);
        channel::Grid      grid(cfg, topo, sub);
        channel::HaloExchanger    halo(topo, sub);
        channel::BoundaryCondition bc(topo, sub);

        channel::MomentumSolver  momentum(cfg, topo, sub, grid, halo);
        channel::PressureSolver  pressure(cfg, topo, sub, grid, halo);
        channel::ChannelForcing  forcing(cfg, topo, sub, grid);
        channel::Statistics      stats(topo, sub, grid);
        channel::RestartIO       restart(topo, sub);
        channel::FieldOutput     field_out(cfg, topo, sub, grid);

        channel::Field<double> U(sub.nx(), sub.ny(), sub.nz());
        channel::Field<double> V(sub.nx(), sub.ny(), sub.nz());
        channel::Field<double> W(sub.nx(), sub.ny(), sub.nz());
        channel::Field<double> P(sub.nx(), sub.ny(), sub.nz());

        channel::RestartState state;
        state.time = cfg.tStart;
        state.dt   = cfg.dtStart;
        state.step = 0;

        if (cfg.ContinueFilein) {
            if (rank == 0)
                std::printf("[main] restarting from %s\n",
                            cfg.dir_cont_filein.c_str());
            restart.read(cfg.dir_cont_filein, U, V, W, P, state);
            forcing.set_mean_dPdx(state.dPdx);
        } else {
            laminar_init(U, V, W, P, sub, grid,
                         cfg.Ly, cfg.Lz, cfg.target_bulk_velocity, cfg.pert_amp,
                         cfg.init_mode, topo.cart());
        }

        halo.exchange(U); halo.exchange(V); halo.exchange(W);
        bc.apply(U, V, W);

        channel::TimeIntegrator integrator(cfg, topo, sub, grid, halo, bc,
                                           momentum, pressure, forcing,
                                           stats, restart, field_out);
        integrator.run(U, V, W, P, state);

        if (rank == 0)
            std::printf("[channel] done. t=%g step=%ld dPdx=%.3e\n",
                        state.time, state.step, state.dPdx);
    }

    MPI_Finalize();
    return rc;
}
