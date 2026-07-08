#include "TimeIntegratorGPU.hpp"
#include "BoundaryConditionGPU.hpp"
#include "ChannelForcingGPU.hpp"
#include "Config.hpp"
#include "FieldOutput.hpp"
#include "Grid.hpp"
#include "HaloExchangerGPU.hpp"
#include "MomentumSolverGPU.hpp"
#include "MpiTopology.hpp"
#include "PressureSolverGPU.hpp"
#include "StatisticsGPU.hpp"
#include "Subdomain.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>
#include <mpi.h>

namespace channel {

namespace gpu_kernels {

__global__ void k_maxdiv_blocks(const double* U, const double* V, const double* W,
                                const double* dx, const double* dy, const double* dz,
                                double* part, int nx, int ny, int nz)
{
    extern __shared__ double sh[];
    int tid = threadIdx.x;
    int n = nx * ny * nz;
    int nxt = nx + 2, nyt = ny + 2;
    double m = 0.0;
    for (int p = blockIdx.x * blockDim.x + tid; p < n; p += blockDim.x * gridDim.x) {
        int i = p % nx + 1, j = (p / nx) % ny + 1, k = p / (nx * ny) + 1;
        double d = fabs((U[df_idx(i+1,j,k,nxt,nyt)] - U[df_idx(i,j,k,nxt,nyt)]) / dx[i]
                      + (V[df_idx(i,j+1,k,nxt,nyt)] - V[df_idx(i,j,k,nxt,nyt)]) / dy[j]
                      + (W[df_idx(i,j,k+1,nxt,nyt)] - W[df_idx(i,j,k,nxt,nyt)]) / dz[k]);
        m = fmax(m, d);
    }
    sh[tid] = m; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] = fmax(sh[tid], sh[tid+s]);
        __syncthreads();
    }
    if (tid == 0) part[blockIdx.x] = sh[0];
}

__global__ void k_cfl_blocks(const double* U, const double* V, const double* W,
                             const double* dx, const double* dy, const double* dz,
                             double* part, int nx, int ny, int nz)
{
    extern __shared__ double sh[];
    int tid = threadIdx.x;
    int n = nx * ny * nz;
    int nxt = nx + 2, nyt = ny + 2;
    double m = 0.0;
    for (int p = blockIdx.x * blockDim.x + tid; p < n; p += blockDim.x * gridDim.x) {
        int i = p % nx + 1, j = (p / nx) % ny + 1, k = p / (nx * ny) + 1;
        double r = fabs(U[df_idx(i,j,k,nxt,nyt)]) / dx[i]
                 + fabs(V[df_idx(i,j,k,nxt,nyt)]) / dy[j]
                 + fabs(W[df_idx(i,j,k,nxt,nyt)]) / dz[k];
        m = fmax(m, isfinite(r) ? r : INFINITY);
    }
    sh[tid] = m; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] = fmax(sh[tid], sh[tid+s]);
        __syncthreads();
    }
    if (tid == 0) part[blockIdx.x] = sh[0];
}

__global__ void k_reduce_max(const double* part, double* out, int n)
{
    extern __shared__ double sh[];
    int tid = threadIdx.x;
    double m = 0.0;
    for (int i = tid; i < n; i += blockDim.x) m = fmax(m, part[i]);
    sh[tid] = m; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] = fmax(sh[tid], sh[tid+s]);
        __syncthreads();
    }
    if (tid == 0) out[0] = sh[0];
}

__global__ void k_reduce_sum(const double* part, double* out, int n)
{
    extern __shared__ double sh[];
    int tid = threadIdx.x;
    double s = 0.0;
    for (int i = tid; i < n; i += blockDim.x) s += part[i];
    sh[tid] = s; __syncthreads();
    for (int q = blockDim.x/2; q > 0; q >>= 1) {
        if (tid < q) sh[tid] += sh[tid+q];
        __syncthreads();
    }
    if (tid == 0) out[0] = sh[0];
}

__global__ void k_wss_blocks(const double* U, const double* dz, double* part,
                             int nx, int ny, int nz, int low, int high, double inv_Re)
{
    extern __shared__ double sh[];
    int tid = threadIdx.x;
    int n = nx * ny;
    int nxt = nx + 2, nyt = ny + 2;
    double s = 0.0;
    for (int p = blockIdx.x * blockDim.x + tid; p < n; p += blockDim.x * gridDim.x) {
        int i = p % nx + 1, j = p / nx + 1;
        if (low != 0) s += U[df_idx(i,j,1,nxt,nyt)] / (0.5 * dz[1]);
        if (high != 0) s += U[df_idx(i,j,nz,nxt,nyt)] / (0.5 * dz[nz]);
    }
    sh[tid] = s; __syncthreads();
    for (int q = blockDim.x/2; q > 0; q >>= 1) {
        if (tid < q) sh[tid] += sh[tid+q];
        __syncthreads();
    }
    if (tid == 0) part[blockIdx.x] = inv_Re * sh[0];
}

} // namespace gpu_kernels
using namespace gpu_kernels;

TimeIntegratorGPU::TimeIntegratorGPU(const Config& cfg, const MpiTopology& topo,
                                     const Subdomain& sub, const Grid& grid,
                                     const HaloExchangerGPU& halo,
                                     const BoundaryConditionGPU& bc,
                                     MomentumSolverGPU& momentum,
                                     PressureSolverGPU& pressure,
                                     ChannelForcingGPU& forcing,
                                     StatisticsGPU& stats,
                                     RestartIO& restart,
                                     FieldOutput& field_out)
    : cfg_(cfg), topo_(topo), sub_(sub), grid_(grid), halo_(halo), bc_(bc),
      momentum_(momentum), pressure_(pressure), forcing_(forcing),
      stats_(stats), restart_(restart), field_out_(field_out) {}

void TimeIntegratorGPU::copy_fields_to_host_(const DeviceField& U, const DeviceField& V,
                                             const DeviceField& W, const DeviceField& P,
                                             Field<double>& hU, Field<double>& hV,
                                             Field<double>& hW, Field<double>& hP) const
{
    U.copy_to_host(hU); V.copy_to_host(hV); W.copy_to_host(hW); P.copy_to_host(hP);
}

double TimeIntegratorGPU::max_div_host_(const DeviceField& U, const DeviceField& V,
                                        const DeviceField& W) const
{
    static DeviceBuffer<double> dx, dy, dz, part, out, global;
    static bool inited = false;
    if (!inited) {
        auto cp = [](DeviceBuffer<double>& d, const std::vector<double>& h) {
            d.reset(h.size());
            CHANNEL_CUDA_CHECK(cudaMemcpy(d.data(), h.data(), h.size()*sizeof(double), cudaMemcpyHostToDevice));
        };
        cp(dx, grid_.dx(0)); cp(dy, grid_.dx(1)); cp(dz, grid_.dx(2));
        part.reset(256); out.reset(1); global.reset(1); inited = true;
    }
    k_maxdiv_blocks<<<256,256,256*sizeof(double)>>>(U.data(), V.data(), W.data(),
                                                    dx.data(), dy.data(), dz.data(), part.data(),
                                                    sub_.nx(), sub_.ny(), sub_.nz());
    // k_maxdiv_blocks yields per-block MAX -> reduce with MAX (not SUM).
    k_reduce_max<<<1,256,256*sizeof(double)>>>(part.data(), out.data(), 256);
    CHANNEL_CUDA_CHECK(cudaGetLastError());
    CHANNEL_CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Allreduce(out.data(), global.data(), 1, MPI_DOUBLE, MPI_MAX, topo_.cart());
    double v = 0.0;
    CHANNEL_CUDA_CHECK(cudaMemcpy(&v, global.data(), sizeof(double), cudaMemcpyDeviceToHost));
    return v;
}

double TimeIntegratorGPU::cfl_dt_host_(const DeviceField& U, const DeviceField& V,
                                       const DeviceField& W) const
{
    static DeviceBuffer<double> dx, dy, dz, part, out, global;
    static bool inited = false;
    if (!inited) {
        auto cp = [](DeviceBuffer<double>& d, const std::vector<double>& h) {
            d.reset(h.size());
            CHANNEL_CUDA_CHECK(cudaMemcpy(d.data(), h.data(), h.size()*sizeof(double), cudaMemcpyHostToDevice));
        };
        cp(dx, grid_.dx(0)); cp(dy, grid_.dx(1)); cp(dz, grid_.dx(2));
        part.reset(256); out.reset(1); global.reset(1); inited = true;
    }
    k_cfl_blocks<<<256,256,256*sizeof(double)>>>(U.data(), V.data(), W.data(),
                                                 dx.data(), dy.data(), dz.data(),
                                                 part.data(), sub_.nx(), sub_.ny(), sub_.nz());
    k_reduce_max<<<1,256,256*sizeof(double)>>>(part.data(), out.data(), 256);
    CHANNEL_CUDA_CHECK(cudaGetLastError());
    CHANNEL_CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Allreduce(out.data(), global.data(), 1, MPI_DOUBLE, MPI_MAX, topo_.cart());
    double global_max = 0.0;
    CHANNEL_CUDA_CHECK(cudaMemcpy(&global_max, global.data(), sizeof(double), cudaMemcpyDeviceToHost));
    if (global_max < 1.0e-30) return cfg_.dtStart;
    return cfg_.MaxCFL / global_max;
}

double TimeIntegratorGPU::wss_host_(const DeviceField& U) const
{
    static DeviceBuffer<double> dz, part, out, global;
    static bool inited = false;
    if (!inited) {
        dz.reset(grid_.dx(2).size());
        CHANNEL_CUDA_CHECK(cudaMemcpy(dz.data(), grid_.dx(2).data(), grid_.dx(2).size()*sizeof(double), cudaMemcpyHostToDevice));
        part.reset(256); out.reset(1); global.reset(1); inited = true;
    }
    bool low = sub_.ista(2) == 1;
    bool high = sub_.ista(2) + sub_.nz() - 1 == sub_.global_n(2);
    k_wss_blocks<<<256,256,256*sizeof(double)>>>(U.data(), dz.data(), part.data(),
                                                 sub_.nx(), sub_.ny(), sub_.nz(),
                                                 low ? 1 : 0, high ? 1 : 0, 1.0 / cfg_.Re_b);
    // k_wss_blocks yields per-block SUM -> reduce with SUM (not MAX).
    k_reduce_sum<<<1,256,256*sizeof(double)>>>(part.data(), out.data(), 256);
    CHANNEL_CUDA_CHECK(cudaGetLastError());
    CHANNEL_CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Allreduce(out.data(), global.data(), 1, MPI_DOUBLE, MPI_SUM, topo_.cart());
    double v = 0.0;
    CHANNEL_CUDA_CHECK(cudaMemcpy(&v, global.data(), sizeof(double), cudaMemcpyDeviceToHost));
    int nxy = sub_.global_n(0) * sub_.global_n(1);
    return std::fabs(v / (2.0 * nxy));
}

std::pair<double,double> TimeIntegratorGPU::rho_diagnostic_(double dt) const
{
    const auto& dz  = grid_.dx(2);
    const auto& dmz = grid_.dmx(2);
    const double nu_h  = 0.5 / cfg_.Re_b;
    const int nz       = sub_.nz();
    const int ista_z   = sub_.ista(2);
    const int n3m      = sub_.global_n(2);
    const int skip     = 2;

    double rho_max_loc = 0.0, rho_min_loc = 1.0;
    for (int k = 1; k <= nz; ++k) {
        int gk = ista_z + k - 1;
        if (gk < skip + 1 || gk > n3m - skip) continue;

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

void TimeIntegratorGPU::run(DeviceField& U, DeviceField& V, DeviceField& W,
                            DeviceField& P, RestartState& state)
{
    const bool root = topo_.rank() == 0;
    double t = state.time;
    double dt = (state.dt > 0.0) ? state.dt : cfg_.dtStart;
    long step = state.step;
    const long step_end = step + cfg_.Timestepmax;
    const int mstr = cfg_.nmonitor > 0 ? cfg_.nmonitor : 1;
    std::FILE* mon_fp = nullptr;
    const bool debug = (std::getenv("CHANNEL_GPU_DEBUG") != nullptr);
    auto debug_div = [&](const char* label, long s) {
        if (!debug) return;
        const double v = max_div_host_(U, V, W);
        if (root) std::printf("[dbg] step=%ld %-15s maxDiv=%.9e\n", s, label, v);
    };
    auto debug_div_bulk = [&](const char* label, long s) {
        if (!debug) return;
        const double v = max_div_host_(U, V, W);
        const double ub = forcing_.bulk_velocity_host(U);
        if (root) std::printf("[dbg] step=%ld %-15s maxDiv=%.9e Ub=%.9e\n", s, label, v, ub);
    };
    std::FILE* wss_fp = nullptr;
    std::FILE* jlog_fp = nullptr;
    if (root) {
        std::string jlog_path = cfg_.dir_statistics + "/J_monitoring.log";
        jlog_fp = std::fopen(jlog_path.c_str(), "a");
        if (jlog_fp && std::ftell(jlog_fp) == 0)
            std::fprintf(jlog_fp, "%10s %6s %6s %6s %6s\n",
                         "step", "rank", "Jx", "Jy", "Jz");

        std::string p = cfg_.dir_instantfield + "Monitor_Channel.plt";
        mon_fp = std::fopen(p.c_str(), "a");
        if (mon_fp && std::ftell(mon_fp) == 0)
            std::fprintf(mon_fp,
                "VARIABLES=\"Timestep\" \"Time\" \"dt\""
                " \"CFL\" \"maxDivU\" \"WSS\" \"u_tau\" \"U_b\"\n"
                "  %12s %12s %12s %12s %12s %12s %12s %12s %12s\n",
                "Timestep", "Time", "dt",
                "maxDivU","WSS","u_tau","U_b","rho_max","rho_min");
        // WSS history file (like CPU reference stats/wss_history.dat)
        std::string wss_path = cfg_.dir_statistics + "/wss_history.dat";
        wss_fp = std::fopen(wss_path.c_str(), "a");
        if (wss_fp && std::ftell(wss_fp) == 0)
            std::fprintf(wss_fp,
                "%12s %12s %12s %12s %12s %12s %12s %12s %12s\n",
                "step", "time", "dt", "wss", "u_tau", "div_max", "U_b",
                "rho_max", "rho_min");
        std::printf("\n%12s %12s %12s %12s %12s %12s %12s %12s %12s\n",
                    "Timestep","Time","dt",
                    "maxDivU","WSS","u_tau","U_b","rho_max","rho_min");
        std::fflush(stdout);
    }

    halo_.exchange(U); halo_.exchange(V); halo_.exchange(W); halo_.exchange(P);
    bc_.apply(U, V, W);

    Field<double> hU(sub_.nx(), sub_.ny(), sub_.nz());
    Field<double> hV(sub_.nx(), sub_.ny(), sub_.nz());
    Field<double> hW(sub_.nx(), sub_.ny(), sub_.nz());
    Field<double> hP(sub_.nx(), sub_.ny(), sub_.nz());

    while (step < step_end) {
        double dt_cfl = cfl_dt_host_(U, V, W);
        dt = std::min(dt_cfl, cfg_.dtStart);
        momentum_.advance(U, V, W, P, dt, forcing_.device_mean_dPdx());
        debug_div("after_momentum", step + 1);
        forcing_.correct(U, dt);
        debug_div_bulk("after_forcing", step + 1);
        halo_.exchange_axis(U, 0);
        halo_.exchange_axis(V, 1);
        halo_.exchange_axis(W, 2);
        bc_.apply(U, V, W);
        debug_div("before_pressure", step + 1);
        pressure_.solve(U, V, W, P, dt);
        debug_div("after_pressure", step + 1);
        halo_.exchange(U); halo_.exchange(V); halo_.exchange(W); halo_.exchange(P);
        bc_.apply(U, V, W);
        debug_div("after_halo_bc", step + 1);

        if (step >= cfg_.nstat_start && ((step - cfg_.nstat_start) % cfg_.nstat == 0))
            stats_.accumulate(U, V, W, P);

        ++step;
        t += dt;

        if (step % mstr == 0) {
            auto [rho_max, rho_min] = rho_diagnostic_(dt);
            double max_div = max_div_host_(U, V, W);
            double wss = wss_host_(U);
            double u_tau = std::sqrt(wss);
            double ub = forcing_.bulk_velocity_host(U);
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
                    max_div, wss, u_tau, ub, rho_max, rho_min);
                std::fflush(stdout);

                if (mon_fp) {
                    std::fprintf(mon_fp,
                        "%d %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n",
                        (int)step, t, dt,
                        max_div, wss, u_tau, ub, rho_max, rho_min);
                    std::fflush(mon_fp);
                }
                if (wss_fp) {
                    std::fprintf(wss_fp,
                        "  %10d %12.6e %12.4e %12.6e %12.6e %12.4e %12.6e %12.6e %12.6e\n",
                        (int)step, t, dt, wss, u_tau, max_div, ub, rho_max, rho_min);
                    std::fflush(wss_fp);
                }
            }

            // Per-rank FilteredTDMA truncation depth (x/y/z), gathered to root.
            int j_loc[3] = { momentum_.last_Jx(), momentum_.last_Jy(), momentum_.last_Jz() };
            int nprocs = topo_.nprocs();
            std::vector<int> j_all(root ? 3 * nprocs : 0);
            MPI_Gather(j_loc, 3, MPI_INT,
                       root ? j_all.data() : nullptr, 3, MPI_INT, 0, topo_.cart());
            if (root && jlog_fp) {
                for (int r = 0; r < nprocs; ++r) {
                    std::fprintf(jlog_fp, "%10ld %6d %6d %6d %6d\n",
                                 step, r, j_all[3*r+0], j_all[3*r+1], j_all[3*r+2]);
                }
                std::fflush(jlog_fp);
            }
        }

        if (cfg_.out_stats && step >= cfg_.nstat_start
                && ((step - cfg_.nstat_start) % cfg_.nout_stats == 0)
                && stats_.samples() > 0) {
            char path[512];
            std::snprintf(path, sizeof(path), "%s/stats_%08ld.dat", cfg_.dir_statistics.c_str(), step);
            stats_.write(path, (int)step, cfg_.Re_b, false);
        }
        if (cfg_.out_field && cfg_.nout > 0 && step >= cfg_.nfield_start
                && ((step - cfg_.nfield_start) % cfg_.nout == 0)) {
            copy_fields_to_host_(U, V, W, P, hU, hV, hW, hP);
            field_out_.write_field_3d(hU, hV, hW, hP, (int)step);
        }
        if (cfg_.ContinueFileout && step >= cfg_.nstat_start
                && ((step - cfg_.nstat_start) % cfg_.nout_stats == 0)) {
            state.time = t; state.dt = dt; state.step = step;
            state.dPdx = forcing_.mean_dPdx_host();
            copy_fields_to_host_(U, V, W, P, hU, hV, hW, hP);
            restart_.write(cfg_.dir_cont_fileout, hU, hV, hW, hP, state, step);
        }
    }

    state.time = t; state.dt = dt; state.step = step;
    state.dPdx = forcing_.mean_dPdx_host();
    if (stats_.samples() > 0) {
        char path[512];
        std::snprintf(path, sizeof(path), "%s/stats_final_%08ld.dat", cfg_.dir_statistics.c_str(), step);
        stats_.write(path, (int)step, cfg_.Re_b, false);
    }
    if (cfg_.ContinueFileout) {
        copy_fields_to_host_(U, V, W, P, hU, hV, hW, hP);
        restart_.write(cfg_.dir_cont_fileout, hU, hV, hW, hP, state, step);
    }
    int my_rank = 0;
    MPI_Comm_rank(topo_.cart(), &my_rank);
    char timing_path[512];
    std::snprintf(timing_path, sizeof(timing_path), "%s/tdma_timing_rank%04d.csv",
                  cfg_.dir_statistics.c_str(), my_rank);
    momentum_.write_timing_csv(timing_path);
    if (root && mon_fp) std::fclose(mon_fp);
    if (root && wss_fp) std::fclose(wss_fp);
    if (root && jlog_fp) std::fclose(jlog_fp);
}

} // namespace channel
