#include <mpi.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <random>

#include "BoundaryConditionGPU.hpp"
#include "ChannelForcingGPU.hpp"
#include "Config.hpp"
#include "DeviceField.hpp"
#include "Field.hpp"
#include "FieldOutput.hpp"
#include "Grid.hpp"
#include "HaloExchangerGPU.hpp"
#include "MomentumSolverGPU.hpp"
#include "MpiTopology.hpp"
#include "PressureSolverGPU.hpp"
#include "RestartIO.hpp"
#include "StatisticsGPU.hpp"
#include "Subdomain.hpp"
#include "TimeIntegratorGPU.hpp"

namespace {

void select_gpu(MPI_Comm comm)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0) MPI_Abort(comm, 1);
    int local_rank = rank;
    if (const char* s = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK")) local_rank = std::atoi(s);
    else if (const char* s = std::getenv("PMI_LOCAL_RANK")) local_rank = std::atoi(s);
    else if (const char* s = std::getenv("SLURM_LOCALID")) local_rank = std::atoi(s);
    cudaSetDevice(local_rank % ndev);
}

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

    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    // Fixed base seed (not std::random_device) so runs are reproducible across
    // hardware/servers for direct comparison; per-rank offset still varies the
    // perturbation spatially.
    constexpr std::uint64_t kBaseSeed = 1234567891ULL;
    std::uint64_t seed = kBaseSeed
                       ^ (static_cast<std::uint64_t>(rank) * 0x9E3779B97F4A7C15ULL);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> rnd(-0.5, 0.5);
    P.fill(0.0);

    auto bulk = [&](const channel::Field<double>& F) {
        double s = 0.0, v = 0.0;
        for (int k = 1; k <= nz; ++k)
            for (int j = 1; j <= ny; ++j)
                for (int i = 1; i <= nx; ++i) {
                    double dV = dx[i] * dy[j] * dz[k];
                    s += F(i,j,k) * dV;
                    v += dV;
                }
        double pkt[2] = {s, v}, tot[2] = {0.0, 0.0};
        MPI_Allreduce(pkt, tot, 2, MPI_DOUBLE, MPI_SUM, comm);
        return (tot[1] > 0.0) ? tot[0] / tot[1] : 0.0;
    };

    if (init_mode == 1) {
        const double nbeta = std::max(1.0, std::round(Ly / Lz));
        const double beta = 2.0 * M_PI * nbeta / Ly;
        const double kz = M_PI / Lz;
        double locmax = 0.0;
        for (int k = 1; k <= nz; ++k)
            for (int j = 1; j <= ny; ++j)
                for (int i = 1; i <= nx; ++i) {
                    double s1 = std::sin(kz * zc[k]);
                    V(i,j,k) = std::cos(beta * yc[j]) * kz * std::sin(2.0 * kz * zc[k]);
                    W(i,j,k) = beta * std::sin(beta * yc[j]) * s1 * s1;
                    locmax = std::max(locmax, std::max(std::fabs(V(i,j,k)), std::fabs(W(i,j,k))));
                }
        double gmax = locmax;
        MPI_Allreduce(MPI_IN_PLACE, &gmax, 1, MPI_DOUBLE, MPI_MAX, comm);
        double scale = (gmax > 0.0) ? (pert * Umax) / gmax : 0.0;
        double noise = 0.1 * pert * Umax;
        for (int k = 1; k <= nz; ++k) {
            double zr = (zc[k] - half) / half;
            double Up = Umax * std::max(0.0, 1.0 - zr*zr);
            for (int j = 1; j <= ny; ++j)
                for (int i = 1; i <= nx; ++i) {
                    U(i,j,k) = Up + noise * rnd(rng);
                    V(i,j,k) = scale * V(i,j,k) + noise * rnd(rng);
                    W(i,j,k) = scale * W(i,j,k) + noise * rnd(rng);
                }
        }
    } else {
        for (int k = 1; k <= nz; ++k)
            for (int j = 1; j <= ny; ++j)
                for (int i = 1; i <= nx; ++i) {
                    U(i,j,k) = rnd(rng);
                    V(i,j,k) = rnd(rng);
                    W(i,j,k) = rnd(rng);
                }
        double Um0 = bulk(U), Vm0 = bulk(V), Wm0 = bulk(W);
        for (int k = 1; k <= nz; ++k)
            for (int j = 1; j <= ny; ++j)
                for (int i = 1; i <= nx; ++i) {
                    U(i,j,k) -= Um0; V(i,j,k) -= Vm0; W(i,j,k) -= Wm0;
                }
        for (int k = 1; k <= nz; ++k) {
            double zr = (zc[k] - half) / half;
            double Up = Umax * std::max(0.0, 1.0 - zr*zr);
            for (int j = 1; j <= ny; ++j)
                for (int i = 1; i <= nx; ++i) {
                    U(i,j,k) = Up + pert * Umax * U(i,j,k);
                    V(i,j,k) =      pert * Umax * V(i,j,k);
                    W(i,j,k) =      pert * Umax * W(i,j,k);
                }
        }
    }

    double Um = bulk(U), Vm = bulk(V), Wm = bulk(W);
    for (int k = 1; k <= nz; ++k)
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i) {
                U(i,j,k) += Ub - Um;
                V(i,j,k) -= Vm;
                W(i,j,k) -= Wm;
            }
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    select_gpu(MPI_COMM_WORLD);

    if (argc < 2) {
        if (rank == 0) std::fprintf(stderr, "usage: %s <PARA_INPUT.dat>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    {
        auto cfg = channel::Config::load(argv[1], MPI_COMM_WORLD);
        if (rank == 0) {
            cfg.print();
            namespace fs = std::filesystem;
            for (const auto& d : { cfg.dir_cont_filein, cfg.dir_cont_fileout,
                                   cfg.dir_instantfield, cfg.dir_statistics }) {
                std::error_code ec;
                fs::create_directories(d, ec);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        channel::MpiTopology topo(MPI_COMM_WORLD, cfg.np1, cfg.np2, cfg.np3,
                                  cfg.pbc1, cfg.pbc2, cfg.pbc3);
        if (rank == 0) topo.print();

        channel::Subdomain sub(cfg, topo);
        channel::Grid grid(cfg, topo, sub);
        channel::HaloExchangerGPU halo(topo, sub);
        channel::BoundaryConditionGPU bc(topo, sub);
        channel::RestartIO restart(topo, sub);
        channel::FieldOutput field_out(cfg, topo, sub, grid);

        channel::Field<double> hU(sub.nx(), sub.ny(), sub.nz());
        channel::Field<double> hV(sub.nx(), sub.ny(), sub.nz());
        channel::Field<double> hW(sub.nx(), sub.ny(), sub.nz());
        channel::Field<double> hP(sub.nx(), sub.ny(), sub.nz());

        channel::RestartState state;
        state.time = cfg.tStart;
        state.dt = cfg.dtStart;
        state.step = 0;

        channel::ChannelForcingGPU forcing(cfg, topo, sub, grid);
        if (cfg.ContinueFilein) {
            restart.read(cfg.dir_cont_filein, hU, hV, hW, hP, state);
            forcing.set_mean_dPdx(state.dPdx);
        } else {
            laminar_init(hU, hV, hW, hP, sub, grid,
                         cfg.Ly, cfg.Lz, cfg.target_bulk_velocity,
                         cfg.pert_amp, cfg.init_mode, topo.cart());
        }

        channel::DeviceField U(sub.nx(), sub.ny(), sub.nz());
        channel::DeviceField V(sub.nx(), sub.ny(), sub.nz());
        channel::DeviceField W(sub.nx(), sub.ny(), sub.nz());
        channel::DeviceField P(sub.nx(), sub.ny(), sub.nz());
        U.copy_from_host(hU); V.copy_from_host(hV); W.copy_from_host(hW); P.copy_from_host(hP);

        channel::MomentumSolverGPU momentum(cfg, topo, sub, grid, halo);
        channel::PressureSolverGPU pressure(cfg, topo, sub, grid, halo);
        channel::StatisticsGPU stats(topo, sub, grid);
        channel::TimeIntegratorGPU integrator(cfg, topo, sub, grid, halo, bc,
                                              momentum, pressure, forcing,
                                              stats, restart, field_out);
        integrator.run(U, V, W, P, state);

        if (rank == 0)
            std::printf("[channel_gpu] done. t=%g step=%ld dPdx=%.3e\n",
                        state.time, state.step, state.dPdx);
    }
    MPI_Finalize();
    return 0;
}
