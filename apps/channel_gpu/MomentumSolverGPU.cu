#include "MomentumSolverGPU.hpp"
#include "Config.hpp"
#include "Grid.hpp"
#include "HaloExchangerGPU.hpp"
#include "MpiTopology.hpp"
#include "Subdomain.hpp"

#include <cstdio>
#include <mpi.h>

namespace channel {

namespace gpu_kernels {

__device__ inline double fld(const double* f, int i, int j, int k, int nxt, int nyt)
{ return f[df_idx(i,j,k,nxt,nyt)]; }

__device__ inline double& fld(double* f, int i, int j, int k, int nxt, int nyt)
{ return f[df_idx(i,j,k,nxt,nyt)]; }

__global__ void k_advection(const double* U, const double* V, const double* W,
                            double* Nu, double* Nv, double* Nw,
                            const double* dx, const double* dmx,
                            const double* dy, const double* dmy,
                            const double* dz, const double* dmz,
                            int nx, int ny, int nz)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nx * ny * nz;
    if (p >= n) return;
    int i = p % nx + 1;
    int j = (p / nx) % ny + 1;
    int k = p / (nx * ny) + 1;
    int nxt = nx + 2, nyt = ny + 2;

    double U_R = 0.5 * (fld(U,i,j,k,nxt,nyt) + fld(U,i+1,j,k,nxt,nyt));
    double U_L = 0.5 * (fld(U,i-1,j,k,nxt,nyt) + fld(U,i,j,k,nxt,nyt));
    double flux_uu = (U_R*U_R - U_L*U_L) / dmx[i];
    double V_top = 0.5 * (fld(V,i-1,j+1,k,nxt,nyt) + fld(V,i,j+1,k,nxt,nyt));
    double V_bot = 0.5 * (fld(V,i-1,j,k,nxt,nyt) + fld(V,i,j,k,nxt,nyt));
    double u_top = 0.5 * (fld(U,i,j,k,nxt,nyt) + fld(U,i,j+1,k,nxt,nyt));
    double u_bot = 0.5 * (fld(U,i,j-1,k,nxt,nyt) + fld(U,i,j,k,nxt,nyt));
    double flux_vu = (V_top*u_top - V_bot*u_bot) / dy[j];
    double W_top = 0.5 * (fld(W,i-1,j,k+1,nxt,nyt) + fld(W,i,j,k+1,nxt,nyt));
    double W_bot = 0.5 * (fld(W,i-1,j,k,nxt,nyt) + fld(W,i,j,k,nxt,nyt));
    double u_zt = 0.5 * (fld(U,i,j,k,nxt,nyt) + fld(U,i,j,k+1,nxt,nyt));
    double u_zb = 0.5 * (fld(U,i,j,k-1,nxt,nyt) + fld(U,i,j,k,nxt,nyt));
    double flux_wu = (W_top*u_zt - W_bot*u_zb) / dz[k];
    fld(Nu,i,j,k,nxt,nyt) = -(flux_uu + flux_vu + flux_wu);

    double UR = 0.5 * (fld(U,i+1,j-1,k,nxt,nyt) + fld(U,i+1,j,k,nxt,nyt));
    double UL = 0.5 * (fld(U,i,j-1,k,nxt,nyt) + fld(U,i,j,k,nxt,nyt));
    double vR = 0.5 * (fld(V,i,j,k,nxt,nyt) + fld(V,i+1,j,k,nxt,nyt));
    double vL = 0.5 * (fld(V,i-1,j,k,nxt,nyt) + fld(V,i,j,k,nxt,nyt));
    double flux_uv = (UR*vR - UL*vL) / dx[i];
    double VT = 0.5 * (fld(V,i,j,k,nxt,nyt) + fld(V,i,j+1,k,nxt,nyt));
    double VB = 0.5 * (fld(V,i,j-1,k,nxt,nyt) + fld(V,i,j,k,nxt,nyt));
    double flux_vv = (VT*VT - VB*VB) / dmy[j];
    double WT = 0.5 * (fld(W,i,j-1,k+1,nxt,nyt) + fld(W,i,j,k+1,nxt,nyt));
    double WB = 0.5 * (fld(W,i,j-1,k,nxt,nyt) + fld(W,i,j,k,nxt,nyt));
    double vzt = 0.5 * (fld(V,i,j,k,nxt,nyt) + fld(V,i,j,k+1,nxt,nyt));
    double vzb = 0.5 * (fld(V,i,j,k-1,nxt,nyt) + fld(V,i,j,k,nxt,nyt));
    double flux_wv = (WT*vzt - WB*vzb) / dz[k];
    fld(Nv,i,j,k,nxt,nyt) = -(flux_uv + flux_vv + flux_wv);

    UR = 0.5 * (fld(U,i+1,j,k-1,nxt,nyt) + fld(U,i+1,j,k,nxt,nyt));
    UL = 0.5 * (fld(U,i,j,k-1,nxt,nyt) + fld(U,i,j,k,nxt,nyt));
    double wR = 0.5 * (fld(W,i,j,k,nxt,nyt) + fld(W,i+1,j,k,nxt,nyt));
    double wL = 0.5 * (fld(W,i-1,j,k,nxt,nyt) + fld(W,i,j,k,nxt,nyt));
    double flux_uw = (UR*wR - UL*wL) / dx[i];
    VT = 0.5 * (fld(V,i,j+1,k-1,nxt,nyt) + fld(V,i,j+1,k,nxt,nyt));
    VB = 0.5 * (fld(V,i,j,k-1,nxt,nyt) + fld(V,i,j,k,nxt,nyt));
    double wT = 0.5 * (fld(W,i,j,k,nxt,nyt) + fld(W,i,j+1,k,nxt,nyt));
    double wB = 0.5 * (fld(W,i,j-1,k,nxt,nyt) + fld(W,i,j,k,nxt,nyt));
    double flux_vw = (VT*wT - VB*wB) / dy[j];
    double Wzt = 0.5 * (fld(W,i,j,k,nxt,nyt) + fld(W,i,j,k+1,nxt,nyt));
    double Wzb = 0.5 * (fld(W,i,j,k-1,nxt,nyt) + fld(W,i,j,k,nxt,nyt));
    double flux_ww = (Wzt*Wzt - Wzb*Wzb) / dmz[k];
    fld(Nw,i,j,k,nxt,nyt) = -(flux_uw + flux_vw + flux_ww);
}

__global__ void k_ab2_body(double* U, double* V, double* W,
                           const double* Nu, const double* Nv, const double* Nw,
                           const double* Nu0, const double* Nv0, const double* Nw0,
                           int nx, int ny, int nz, double dt, int first,
                           const double* mean_dPdx)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nx * ny * nz;
    if (p >= n) return;
    int i = p % nx + 1;
    int j = (p / nx) % ny + 1;
    int k = p / (nx * ny) + 1;
    int nxt = nx + 2, nyt = ny + 2;
    double au = fld(Nu,i,j,k,nxt,nyt), av = fld(Nv,i,j,k,nxt,nyt), aw = fld(Nw,i,j,k,nxt,nyt);
    if (first == 0) {
        au = 1.5 * au - 0.5 * fld(Nu0,i,j,k,nxt,nyt);
        av = 1.5 * av - 0.5 * fld(Nv0,i,j,k,nxt,nyt);
        aw = 1.5 * aw - 0.5 * fld(Nw0,i,j,k,nxt,nyt);
    }
    fld(U,i,j,k,nxt,nyt) += dt * (au - *mean_dPdx);
    fld(V,i,j,k,nxt,nyt) += dt * av;
    fld(W,i,j,k,nxt,nyt) += dt * aw;
}

__global__ void k_lap(int comp, const double* q, double* rhs,
                      const double* dx, const double* dmx,
                      const double* dy, const double* dmy,
                      const double* dz, const double* dmz,
                      int nx, int ny, int nz, double c)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nx * ny * nz;
    if (p >= n) return;
    int i = p % nx + 1;
    int j = (p / nx) % ny + 1;
    int k = p / (nx * ny) + 1;
    int nxt = nx + 2, nyt = ny + 2;
    double ax, bx, cx, ay, by, cy, az, bz, cz;
    if (comp == 0) { ax = 1.0/(dmx[i]*dx[i-1]); cx = 1.0/(dmx[i]*dx[i]); }
    else           { ax = 1.0/(dx[i]*dmx[i]);   cx = 1.0/(dx[i]*dmx[i+1]); }
    bx = -(ax + cx);
    if (comp == 1) { ay = 1.0/(dmy[j]*dy[j-1]); cy = 1.0/(dmy[j]*dy[j]); }
    else           { ay = 1.0/(dy[j]*dmy[j]);   cy = 1.0/(dy[j]*dmy[j+1]); }
    by = -(ay + cy);
    if (comp == 2) { az = 1.0/(dmz[k]*dz[k-1]); cz = 1.0/(dmz[k]*dz[k]); }
    else           { az = 1.0/(dz[k]*dmz[k]);   cz = 1.0/(dz[k]*dmz[k+1]); }
    bz = -(az + cz);
    double lap = ax*fld(q,i-1,j,k,nxt,nyt) + bx*fld(q,i,j,k,nxt,nyt) + cx*fld(q,i+1,j,k,nxt,nyt)
               + ay*fld(q,i,j-1,k,nxt,nyt) + by*fld(q,i,j,k,nxt,nyt) + cy*fld(q,i,j+1,k,nxt,nyt)
               + az*fld(q,i,j,k-1,nxt,nyt) + bz*fld(q,i,j,k,nxt,nyt) + cz*fld(q,i,j,k+1,nxt,nyt);
    fld(rhs,i,j,k,nxt,nyt) = fld(q,i,j,k,nxt,nyt) + c * lap;
}

__global__ void k_build_z(int comp, const double* q, double* A, double* B, double* C, double* D,
                          const double* dz, const double* dmz,
                          int nx, int ny, int nz, double c, int low, int high)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int ns = nx * ny, n = ns * nz;
    if (p >= n) return;
    int s = p % ns;
    int row = p / ns;
    int i = s / ny + 1;
    int j = s % ny + 1;
    int k = row + 1;
    bool is_w = comp == 2;
    double ar = is_w ? 1.0/(dmz[k]*dz[k-1]) : 1.0/(dz[k]*dmz[k]);
    double cr = is_w ? 1.0/(dmz[k]*dz[k])   : 1.0/(dz[k]*dmz[k+1]);
    double a = -c * ar, cc = -c * cr, b = 1.0 - a - cc;
    if (low != 0 && k == 1) { a = 0.0; if (!is_w) b += c * ar; }
    if (high != 0 && k == nz) { cc = 0.0; if (!is_w) b += c * cr; }
    if (low != 0 && is_w && k == 1) { a = 0.0; b = 1.0; cc = 0.0; }
    int idx = row * ns + s;
    A[idx] = a; B[idx] = b; C[idx] = cc;
    D[idx] = (low != 0 && is_w && k == 1) ? 0.0 : fld(q,i,j,k,nx+2,ny+2);
}

__global__ void k_scatter_z(const double* D, double* q, int nx, int ny, int nz)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int ns = nx * ny, n = ns * nz;
    if (p >= n) return;
    int s = p % ns, row = p / ns;
    int i = s / ny + 1, j = s % ny + 1, k = row + 1;
    fld(q,i,j,k,nx+2,ny+2) = D[p];
}

__global__ void k_build_y(int comp, const double* q, double* A, double* B, double* C, double* D,
                          const double* dy, const double* dmy,
                          int nx, int ny, int nz, double c)
{
    int ns = nx * nz;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int n = ns * ny;
    if (p >= n) return;
    int s = p % ns, row = p / ns;
    int i = s / nz + 1, k = s % nz + 1, j = row + 1;
    double ar = (comp == 1) ? 1.0/(dmy[j]*dy[j-1]) : 1.0/(dy[j]*dmy[j]);
    double cr = (comp == 1) ? 1.0/(dmy[j]*dy[j])   : 1.0/(dy[j]*dmy[j+1]);
    double a = -c * ar, cc = -c * cr, b = 1.0 - a - cc;
    A[p] = a; B[p] = b; C[p] = cc; D[p] = fld(q,i,j,k,nx+2,ny+2);
}

__global__ void k_scatter_y(const double* D, double* q, int nx, int ny, int nz)
{
    int ns = nx * nz;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int n = ns * ny;
    if (p >= n) return;
    int s = p % ns, row = p / ns;
    int i = s / nz + 1, k = s % nz + 1, j = row + 1;
    fld(q,i,j,k,nx+2,ny+2) = D[p];
}

__global__ void k_build_x(int comp, const double* q, double* A, double* B, double* C, double* D,
                          const double* dx, const double* dmx,
                          int nx, int ny, int nz, double c)
{
    int ns = ny * nz;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int n = ns * nx;
    if (p >= n) return;
    int s = p % ns, row = p / ns;
    int j = s / nz + 1, k = s % nz + 1, i = row + 1;
    double ar = (comp == 0) ? 1.0/(dmx[i]*dx[i-1]) : 1.0/(dx[i]*dmx[i]);
    double cr = (comp == 0) ? 1.0/(dmx[i]*dx[i])   : 1.0/(dx[i]*dmx[i+1]);
    double a = -c * ar, cc = -c * cr, b = 1.0 - a - cc;
    A[p] = a; B[p] = b; C[p] = cc; D[p] = fld(q,i,j,k,nx+2,ny+2);
}

__global__ void k_scatter_x(const double* D, double* out, int nx, int ny, int nz)
{
    int ns = ny * nz;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int n = ns * nx;
    if (p >= n) return;
    int s = p % ns, row = p / ns;
    int j = s / nz + 1, k = s % nz + 1, i = row + 1;
    fld(out,i,j,k,nx+2,ny+2) = D[p];
}

} // namespace gpu_kernels
using namespace gpu_kernels;

static void copy_metric(DeviceBuffer<double>& dst, const std::vector<double>& src)
{
    dst.reset(src.size());
    CHANNEL_CUDA_CHECK(cudaMemcpy(dst.data(), src.data(), src.size() * sizeof(double),
                                  cudaMemcpyHostToDevice));
}

MomentumSolverGPU::MomentumSolverGPU(const Config& cfg, const MpiTopology& topo,
                                     const Subdomain& sub, const Grid& grid,
                                     const HaloExchangerGPU& halo)
    : cfg_(cfg), halo_(halo),
      nx_(sub.nx()), ny_(sub.ny()), nz_(sub.nz()),
      np3_(topo.dim(2)), rank_z_(topo.rank_in(2)),
      inv_Re_(1.0 / cfg.Re_b),
      Nu_old_(nx_,ny_,nz_), Nv_old_(nx_,ny_,nz_), Nw_old_(nx_,ny_,nz_),
      Nu_new_(nx_,ny_,nz_), Nv_new_(nx_,ny_,nz_), Nw_new_(nx_,ny_,nz_),
      rhs_(nx_,ny_,nz_)
{
    for (int a = 0; a < 3; ++a) {
        copy_metric(dx_[a], grid.dx(a));
        copy_metric(dmx_[a], grid.dmx(a));
    }
    Ax_.reset(static_cast<std::size_t>(nx_) * ny_ * nz_);
    Bx_.reset(Ax_.size()); Cx_.reset(Ax_.size()); Dx_.reset(Ax_.size());
    Ay_.reset(static_cast<std::size_t>(ny_) * nx_ * nz_);
    By_.reset(Ay_.size()); Cy_.reset(Ay_.size()); Dy_.reset(Ay_.size());
    Az_.reset(static_cast<std::size_t>(nz_) * nx_ * ny_);
    Bz_.reset(Az_.size()); Cz_.reset(Az_.size()); Dz_.reset(Az_.size());

    auto backend = TdmaSolverGPU::parse_backend(cfg.tdma_backend);
    fdma_x_ = std::make_unique<TdmaSolverGPU>(topo, 0, ny_*nz_, nx_, cfg.pbc1, backend, 1.0e-12);
    fdma_y_ = std::make_unique<TdmaSolverGPU>(topo, 1, nx_*nz_, ny_, cfg.pbc2, backend, 1.0e-12);
    fdma_z_ = std::make_unique<TdmaSolverGPU>(topo, 2, nx_*ny_, nz_, false, backend, 1.0e-12);
}

void MomentumSolverGPU::adi_z_(Component c, DeviceField& q, double nu_dt_half)
{
    int n = nx_ * ny_ * nz_;
    int block = 256, grid = (n + block - 1) / block;
    double t0 = MPI_Wtime();
    k_build_z<<<grid, block>>>(static_cast<int>(c), q.data(), Az_.data(), Bz_.data(), Cz_.data(),
                               Dz_.data(), dx_[2].data(), dmx_[2].data(),
                               nx_, ny_, nz_, nu_dt_half,
                               rank_z_ == 0 ? 1 : 0, rank_z_ == np3_ - 1 ? 1 : 0);
    CHANNEL_CUDA_CHECK(cudaGetLastError());
    fdma_z_->set_rho_device(Az_.data(), Bz_.data(), Cz_.data());
    fdma_z_->solve(Az_.data(), Bz_.data(), Cz_.data(), Dz_.data());
    k_scatter_z<<<grid, block>>>(Dz_.data(), q.data(), nx_, ny_, nz_);
    CHANNEL_CUDA_CHECK(cudaGetLastError());
    CHANNEL_CUDA_CHECK(cudaDeviceSynchronize());
    if (step_count_ > cfg_.nstat_start) tdma_z_time_ += MPI_Wtime() - t0;
}

void MomentumSolverGPU::adi_y_(Component c, DeviceField& q, double nu_dt_half)
{
    int n = nx_ * ny_ * nz_;
    int block = 256, grid = (n + block - 1) / block;
    double t0 = MPI_Wtime();
    k_build_y<<<grid, block>>>(static_cast<int>(c), q.data(), Ay_.data(), By_.data(), Cy_.data(),
                               Dy_.data(), dx_[1].data(), dmx_[1].data(),
                               nx_, ny_, nz_, nu_dt_half);
    CHANNEL_CUDA_CHECK(cudaGetLastError());
    fdma_y_->set_rho_device(Ay_.data(), By_.data(), Cy_.data());
    if (cfg_.pbc2) fdma_y_->solve_cycl(Ay_.data(), By_.data(), Cy_.data(), Dy_.data());
    else fdma_y_->solve(Ay_.data(), By_.data(), Cy_.data(), Dy_.data());
    k_scatter_y<<<grid, block>>>(Dy_.data(), q.data(), nx_, ny_, nz_);
    CHANNEL_CUDA_CHECK(cudaGetLastError());
    CHANNEL_CUDA_CHECK(cudaDeviceSynchronize());
    if (step_count_ > cfg_.nstat_start) tdma_y_time_ += MPI_Wtime() - t0;
}

void MomentumSolverGPU::adi_x_(Component c, DeviceField& q, double nu_dt_half, DeviceField& dst)
{
    int n = nx_ * ny_ * nz_;
    int block = 256, grid = (n + block - 1) / block;
    double t0 = MPI_Wtime();
    k_build_x<<<grid, block>>>(static_cast<int>(c), q.data(), Ax_.data(), Bx_.data(), Cx_.data(),
                               Dx_.data(), dx_[0].data(), dmx_[0].data(),
                               nx_, ny_, nz_, nu_dt_half);
    CHANNEL_CUDA_CHECK(cudaGetLastError());
    fdma_x_->set_rho_device(Ax_.data(), Bx_.data(), Cx_.data());
    if (cfg_.pbc1) fdma_x_->solve_cycl(Ax_.data(), Bx_.data(), Cx_.data(), Dx_.data());
    else fdma_x_->solve(Ax_.data(), Bx_.data(), Cx_.data(), Dx_.data());
    k_scatter_x<<<grid, block>>>(Dx_.data(), dst.data(), nx_, ny_, nz_);
    CHANNEL_CUDA_CHECK(cudaGetLastError());
    CHANNEL_CUDA_CHECK(cudaDeviceSynchronize());
    if (step_count_ > cfg_.nstat_start) tdma_x_time_ += MPI_Wtime() - t0;
}

void MomentumSolverGPU::advance(DeviceField& U, DeviceField& V, DeviceField& W,
                                const DeviceField&, double dt, const double* d_mean_dPdx)
{
    ++step_count_;
    double t0 = MPI_Wtime();
    int n = nx_ * ny_ * nz_;
    int block = 256, grid = (n + block - 1) / block;
    fdma_x_->set_eps_constant(dt);
    fdma_y_->set_eps_constant(dt);
    fdma_z_->set_eps_constant(dt);

    k_advection<<<grid, block>>>(U.data(), V.data(), W.data(),
                                 Nu_new_.data(), Nv_new_.data(), Nw_new_.data(),
                                 dx_[0].data(), dmx_[0].data(),
                                 dx_[1].data(), dmx_[1].data(),
                                 dx_[2].data(), dmx_[2].data(),
                                 nx_, ny_, nz_);
    CHANNEL_CUDA_CHECK(cudaGetLastError());
    k_ab2_body<<<grid, block>>>(U.data(), V.data(), W.data(),
                                Nu_new_.data(), Nv_new_.data(), Nw_new_.data(),
                                Nu_old_.data(), Nv_old_.data(), Nw_old_.data(),
                                nx_, ny_, nz_, dt, first_step_ ? 1 : 0, d_mean_dPdx);
    CHANNEL_CUDA_CHECK(cudaGetLastError());
    halo_.exchange(U); halo_.exchange(V); halo_.exchange(W);

    const double nu_dt_half = 0.5 * inv_Re_ * dt;
    auto diffuse = [&](Component c, DeviceField& q) {
        k_lap<<<grid, block>>>(static_cast<int>(c), q.data(), rhs_.data(),
                               dx_[0].data(), dmx_[0].data(),
                               dx_[1].data(), dmx_[1].data(),
                               dx_[2].data(), dmx_[2].data(),
                               nx_, ny_, nz_, nu_dt_half);
        CHANNEL_CUDA_CHECK(cudaGetLastError());
        adi_z_(c, rhs_, nu_dt_half);
        adi_y_(c, rhs_, nu_dt_half);
        adi_x_(c, rhs_, nu_dt_half, q);
    };
    diffuse(COMP_U, U);
    diffuse(COMP_V, V);
    diffuse(COMP_W, W);

    Nu_old_.swap(Nu_new_);
    Nv_old_.swap(Nv_new_);
    Nw_old_.swap(Nw_new_);
    first_step_ = false;
    if (step_count_ > cfg_.nstat_start) {
        momentum_time_ += MPI_Wtime() - t0;
        if ((step_count_ - cfg_.nstat_start) % 100 == 0) {
            timing_step_.push_back(step_count_);
            timing_z_.push_back(tdma_z_time_);
            timing_y_.push_back(tdma_y_time_);
            timing_x_.push_back(tdma_x_time_);
            timing_mom_.push_back(momentum_time_);
        }
    }
}

void MomentumSolverGPU::write_timing_csv(const std::string& path) const
{
    std::FILE* fp = std::fopen(path.c_str(), "w");
    if (!fp) return;
    std::fprintf(fp, "# cumulative timings start when step > nstat_start (%d)\n", cfg_.nstat_start);
    std::fprintf(fp, "# tdma_* columns include coefficient/RHS build, TDMA solve, scatter, and device synchronization for that ADI direction\n");
    std::fprintf(fp, "step,timed_steps,tdma_z_sec_cum,tdma_y_sec_cum,tdma_x_sec_cum,tdma_total_sec_cum,momentum_sec_cum,tdma_over_momentum\n");
    for (std::size_t i = 0; i < timing_step_.size(); ++i) {
        double tdma_total = timing_z_[i] + timing_y_[i] + timing_x_[i];
        double ratio = timing_mom_[i] > 0.0 ? tdma_total / timing_mom_[i] : 0.0;
        std::fprintf(fp, "%ld,%ld,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e\n",
                     timing_step_[i], timing_step_[i] - cfg_.nstat_start,
                     timing_z_[i], timing_y_[i], timing_x_[i],
                     tdma_total, timing_mom_[i], ratio);
    }
    const bool aligned = !timing_step_.empty() && timing_step_.back() == step_count_;
    if (step_count_ > cfg_.nstat_start && !aligned) {
        double tdma_total = tdma_z_time_ + tdma_y_time_ + tdma_x_time_;
        double ratio = momentum_time_ > 0.0 ? tdma_total / momentum_time_ : 0.0;
        std::fprintf(fp, "%ld,%ld,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e\n",
                     step_count_, step_count_ - cfg_.nstat_start,
                     tdma_z_time_, tdma_y_time_, tdma_x_time_,
                     tdma_total, momentum_time_, ratio);
    }
    std::fclose(fp);
}

} // namespace channel
