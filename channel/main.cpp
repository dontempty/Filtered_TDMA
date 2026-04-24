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

void laminar_init(channel::Field<double>& U, channel::Field<double>& V,
                  channel::Field<double>& W, channel::Field<double>& P,
                  const channel::Subdomain& sub, const channel::Grid& grid,
                  double Lz, double Ub, double pert)
{
    const int nx = sub.nx(), ny = sub.ny(), nz = sub.nz();
    const auto& zc = grid.x(2);
    const double Umax = 1.5 * Ub;
    const double half = 0.5 * Lz;

    std::mt19937 rng(12345 + sub.ista(2));
    std::uniform_real_distribution<double> u(-1.0, 1.0);

    P.fill(0.0);
    V.fill(0.0);
    W.fill(0.0);
    for (int k = 1; k <= nz; ++k) {
        double zr = (zc[k] - half) / half;
        double Up = Umax * std::max(0.0, 1.0 - zr * zr);
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i)
                U(i, j, k) = Up + pert * Umax * u(rng);
    }
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

        channel::MomentumSolver  momentum(cfg, topo, sub, grid);
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
                         cfg.Lz, cfg.target_bulk_velocity, /*pert=*/0.05);
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
