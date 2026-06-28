#include "PressureSolverGPU.hpp"
#include "Config.hpp"
#include "Grid.hpp"
#include "HaloExchangerGPU.hpp"
#include "MpiTopology.hpp"
#include "Subdomain.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace channel {

namespace gpu_kernels {

inline void cufft_check(cufftResult r, const char* expr, const char* file, int line)
{
    if (r != CUFFT_SUCCESS) {
        std::fprintf(stderr, "[cuFFT] %s failed at %s:%d: %d\n", expr, file, line, (int)r);
        std::abort();
    }
}

#define CUFFT_CHECK(expr) cufft_check((expr), #expr, __FILE__, __LINE__)

__device__ inline std::size_t idxC(int i, int j, int k, int nx, int ny)
{ return static_cast<std::size_t>(i) + static_cast<std::size_t>(nx) * (j + static_cast<std::size_t>(ny) * k); }

__device__ inline std::size_t idxI(int i, int j, int k, int nx, int ny)
{ return static_cast<std::size_t>(i) + static_cast<std::size_t>(nx) * (j + static_cast<std::size_t>(ny) * k); }

__device__ inline std::size_t idxY(int ix, int j, int k, int nxh, int ny)
{ return static_cast<std::size_t>(ix) + static_cast<std::size_t>(nxh) * (j + static_cast<std::size_t>(ny) * k); }

__global__ void k_rhs(const double* U, const double* V, const double* W, double* rhs,
                      const double* dx, const double* dy, const double* dz,
                      int nx, int ny, int nz, double inv_dt)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nx * ny * nz;
    if (p >= n) return;
    int i = p % nx + 1, j = (p / nx) % ny + 1, k = p / (nx * ny) + 1;
    int nxt = nx + 2, nyt = ny + 2;
    double d = (U[df_idx(i+1,j,k,nxt,nyt)] - U[df_idx(i,j,k,nxt,nyt)]) / dx[i]
             + (V[df_idx(i,j+1,k,nxt,nyt)] - V[df_idx(i,j,k,nxt,nyt)]) / dy[j]
             + (W[df_idx(i,j,k+1,nxt,nyt)] - W[df_idx(i,j,k,nxt,nyt)]) / dz[k];
    rhs[idxC(i-1,j-1,k-1,nx,ny)] = d * inv_dt;
}

__global__ void k_pack_C_to_I(const double* rhs_C, double* sbuf,
                              int nx, int ny, int nz, int n3I, int np1)
{
    int blk = nx * ny * n3I;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blk * np1;
    if (p >= n) return;
    int s = p / blk, q = p % blk;
    int i = q / (ny * n3I);
    int r = q % (ny * n3I);
    int j = r / n3I;
    int k = r % n3I;
    sbuf[p] = rhs_C[idxC(i, j, s*n3I + k, nx, ny)];
}

__global__ void k_unpack_C_to_I(const double* rbuf, double* rhs_I,
                                int nx, int ny, int n1m, int n3I, int np1)
{
    int blk = nx * ny * n3I;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blk * np1;
    if (p >= n) return;
    int s = p / blk, q = p % blk;
    int i = q / (ny * n3I);
    int r = q % (ny * n3I);
    int j = r / n3I;
    int k = r % n3I;
    rhs_I[idxI(s*nx + i, j, k, n1m, ny)] = rbuf[p];
}

__global__ void k_pack_I_to_C(const double* dp_I, double* sbuf,
                              int nx, int ny, int n1m, int n3I, int np1)
{
    int blk = nx * ny * n3I;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blk * np1;
    if (p >= n) return;
    int s = p / blk, q = p % blk;
    int i = q / (ny * n3I);
    int r = q % (ny * n3I);
    int j = r / n3I;
    int k = r % n3I;
    sbuf[p] = dp_I[idxI(s*nx + i, j, k, n1m, ny)];
}

__global__ void k_unpack_I_to_C(const double* rbuf, double* dp,
                                int nx, int ny, int nz, int n3I, int np1)
{
    int blk = nx * ny * n3I;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blk * np1;
    if (p >= n) return;
    int s = p / blk, q = p % blk;
    int i = q / (ny * n3I);
    int r = q % (ny * n3I);
    int j = r / n3I;
    int k = r % n3I;
    int nxt = nx + 2, nyt = ny + 2;
    dp[df_idx(i+1, j+1, s*n3I + k + 1, nxt, nyt)] = rbuf[p];
}

__global__ void k_pack_I_to_Y(const cufftDoubleComplex* hat_I, double* sbuf,
                              int rdest, int ix0, int nxr, int off,
                              int Nxh, int nyloc, int n3I)
{
    int yn3 = nyloc * n3I;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nxr * yn3;
    if (p >= n) return;
    int ixl = p / yn3;
    int rem = p % yn3;
    int j = rem / n3I;
    int k = rem % n3I;
    cufftDoubleComplex z = hat_I[idxI(ix0 + ixl, j, k, Nxh, nyloc)];
    int q = off + p * 2;
    sbuf[q] = z.x; sbuf[q+1] = z.y;
    (void)rdest;
}

__global__ void k_unpack_I_to_Y(const double* rbuf, cufftDoubleComplex* hat_Y,
                                int rsrc, int off, int hme,
                                int nyloc, int n2m, int n3I)
{
    int yn3 = nyloc * n3I;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int n = hme * yn3;
    if (p >= n) return;
    int ixl = p / yn3;
    int rem = p % yn3;
    int jl = rem / n3I;
    int k = rem % n3I;
    int q = off + p * 2;
    int j = rsrc * nyloc + jl;
    hat_Y[idxY(ixl, j, k, hme, n2m)] = make_cuDoubleComplex(rbuf[q], rbuf[q+1]);
}

__global__ void k_pack_Y_to_I(const cufftDoubleComplex* hat_Y, double* sbuf,
                              int rdest, int off, int hme,
                              int nyloc, int n2m, int n3I)
{
    int yn3 = nyloc * n3I;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int n = hme * yn3;
    if (p >= n) return;
    int ixl = p / yn3;
    int rem = p % yn3;
    int jl = rem / n3I;
    int k = rem % n3I;
    int j = rdest * nyloc + jl;
    cufftDoubleComplex z = hat_Y[idxY(ixl, j, k, hme, n2m)];
    int q = off + p * 2;
    sbuf[q] = z.x; sbuf[q+1] = z.y;
}

__global__ void k_unpack_Y_to_I(const double* rbuf, cufftDoubleComplex* hat_I,
                                int rsrc, int ix0, int nxr, int off,
                                int Nxh, int nyloc, int n3I)
{
    int yn3 = nyloc * n3I;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nxr * yn3;
    if (p >= n) return;
    int ixl = p / yn3;
    int rem = p % yn3;
    int j = rem / n3I;
    int k = rem % n3I;
    int q = off + p * 2;
    hat_I[idxI(ix0 + ixl, j, k, Nxh, nyloc)] = make_cuDoubleComplex(rbuf[q], rbuf[q+1]);
    (void)rsrc;
}

__global__ void k_build_poisson_tdma(const cufftDoubleComplex* hat_Y,
                                     double* Am, double* Ac, double* Ap,
                                     double* Am2, double* Ac2, double* Ap2,
                                     double* Br, double* Bi,
                                     const double* dzg, const double* dmzg,
                                     const double* kx2, const double* ky2,
                                     int n3I, int nsys, int hme, int n2m,
                                     int ix0, int rank_xz, int size_xz)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int n = n3I * nsys;
    if (p >= n) return;
    int s = p % nsys;
    int k = p / nsys;
    int ixl = s % hme;
    int j = s / hme;
    int kg = rank_xz * n3I + k;
    double dz = dzg[kg + 1], dmz = dmzg[kg + 1], dmzp = dmzg[kg + 2];
    bool lower = (rank_xz == 0 && k == 0);
    bool upper = (rank_xz == size_xz - 1 && k == n3I - 1);
    double a = lower ? 0.0 : 1.0 / (dz * dmz);
    double c = upper ? 0.0 : 1.0 / (dz * dmzp);
    double lambda = kx2[ix0 + ixl] + ky2[j];
    double b = -(a + c) - lambda;
    Am[p] = Am2[p] = a; Ac[p] = Ac2[p] = b; Ap[p] = Ap2[p] = c;
    cufftDoubleComplex z = hat_Y[idxY(ixl, j, k, hme, n2m)];
    Br[p] = z.x; Bi[p] = z.y;
}

__global__ void k_zero_mode(double* Am, double* Ac, double* Ap,
                            double* Am2, double* Ac2, double* Ap2,
                            double* Br, double* Bi)
{
    Am[0] = Am2[0] = 0.0;
    Ac[0] = Ac2[0] = 1.0;
    Ap[0] = Ap2[0] = 0.0;
    Br[0] = 0.0; Bi[0] = 0.0;
}

__global__ void k_write_poisson_solution(cufftDoubleComplex* hat_Y,
                                         const double* Br, const double* Bi,
                                         int n3I, int nsys, int hme, int n2m)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int n = n3I * nsys;
    if (p >= n) return;
    int s = p % nsys;
    int k = p / nsys;
    int ixl = s % hme;
    int j = s / hme;
    hat_Y[idxY(ixl, j, k, hme, n2m)] = make_cuDoubleComplex(Br[p], Bi[p]);
}

__global__ void k_scale(double* p, std::size_t n, double scale)
{
    std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t stride = blockDim.x * gridDim.x;
    for (std::size_t q = tid; q < n; q += stride) p[q] *= scale;
}

__global__ void k_poisson_residual_blocks(const double* dp, const double* rhs,
                                          const double* dx, const double* dy, const double* dz,
                                          const double* dmx, const double* dmy, const double* dmz,
                                          double* part_rhs, double* part_lap, double* part_res,
                                          int nx, int ny, int nz)
{
    extern __shared__ double sh[];
    double* sh_rhs = sh;
    double* sh_lap = sh + blockDim.x;
    double* sh_res = sh + 2 * blockDim.x;
    int tid = threadIdx.x;
    int n = nx * ny * nz;
    int nxt = nx + 2, nyt = ny + 2;
    double mr = 0.0, ml = 0.0, me = 0.0;
    for (int p = blockIdx.x * blockDim.x + tid; p < n; p += blockDim.x * gridDim.x) {
        int i = p % nx + 1, j = (p / nx) % ny + 1, k = p / (nx * ny) + 1;
        double pc = dp[df_idx(i,j,k,nxt,nyt)];
        double lap =
            ((dp[df_idx(i+1,j,k,nxt,nyt)] - pc) / dmx[i+1]
           - (pc - dp[df_idx(i-1,j,k,nxt,nyt)]) / dmx[i]) / dx[i]
          + ((dp[df_idx(i,j+1,k,nxt,nyt)] - pc) / dmy[j+1]
           - (pc - dp[df_idx(i,j-1,k,nxt,nyt)]) / dmy[j]) / dy[j]
          + ((dp[df_idx(i,j,k+1,nxt,nyt)] - pc) / dmz[k+1]
           - (pc - dp[df_idx(i,j,k-1,nxt,nyt)]) / dmz[k]) / dz[k];
        double b = rhs[idxC(i-1,j-1,k-1,nx,ny)];
        double e = lap - b;
        mr = fmax(mr, isfinite(b) ? fabs(b) : INFINITY);
        ml = fmax(ml, isfinite(lap) ? fabs(lap) : INFINITY);
        me = fmax(me, isfinite(e) ? fabs(e) : INFINITY);
    }
    sh_rhs[tid] = mr;
    sh_lap[tid] = ml;
    sh_res[tid] = me;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh_rhs[tid] = fmax(sh_rhs[tid], sh_rhs[tid+s]);
            sh_lap[tid] = fmax(sh_lap[tid], sh_lap[tid+s]);
            sh_res[tid] = fmax(sh_res[tid], sh_res[tid+s]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        part_rhs[blockIdx.x] = sh_rhs[0];
        part_lap[blockIdx.x] = sh_lap[0];
        part_res[blockIdx.x] = sh_res[0];
    }
}

__global__ void k_dp_wall(double* dp, int nx, int ny, int nz, int low, int high)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nx * ny;
    if (p >= n) return;
    int i = p % nx + 1, j = p / nx + 1;
    int nxt = nx + 2, nyt = ny + 2;
    if (low != 0) dp[df_idx(i,j,0,nxt,nyt)] = dp[df_idx(i,j,1,nxt,nyt)];
    if (high != 0) dp[df_idx(i,j,nz+1,nxt,nyt)] = dp[df_idx(i,j,nz,nxt,nyt)];
}

__global__ void k_project(double* U, double* V, double* W, double* P, const double* dp,
                          const double* dmx, const double* dmy, const double* dmz,
                          int nx, int ny, int nz, double dt)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nx * ny * nz;
    if (p >= n) return;
    int i = p % nx + 1, j = (p / nx) % ny + 1, k = p / (nx * ny) + 1;
    int nxt = nx + 2, nyt = ny + 2;
    P[df_idx(i,j,k,nxt,nyt)] += dp[df_idx(i,j,k,nxt,nyt)];
    U[df_idx(i,j,k,nxt,nyt)] -= dt * (dp[df_idx(i,j,k,nxt,nyt)] - dp[df_idx(i-1,j,k,nxt,nyt)]) / dmx[i];
    V[df_idx(i,j,k,nxt,nyt)] -= dt * (dp[df_idx(i,j,k,nxt,nyt)] - dp[df_idx(i,j-1,k,nxt,nyt)]) / dmy[j];
    W[df_idx(i,j,k,nxt,nyt)] -= dt * (dp[df_idx(i,j,k,nxt,nyt)] - dp[df_idx(i,j,k-1,nxt,nyt)]) / dmz[k];
}

} // namespace gpu_kernels
using namespace gpu_kernels;

static void copy_vec(DeviceBuffer<double>& d, const std::vector<double>& h)
{
    d.reset(h.size());
    CHANNEL_CUDA_CHECK(cudaMemcpy(d.data(), h.data(), h.size()*sizeof(double), cudaMemcpyHostToDevice));
}

PressureSolverGPU::PressureSolverGPU(const Config& cfg, const MpiTopology& topo,
                                     const Subdomain& sub, const Grid& grid,
                                     const HaloExchangerGPU& halo)
    : topo_(topo), halo_(halo), dp_(sub.nx(), sub.ny(), sub.nz())
{
    np1_ = topo.dim(0); np2_ = topo.dim(1); np3_ = topo.dim(2);
    nx_loc_ = sub.nx(); ny_loc_ = sub.ny(); nz_loc_ = sub.nz();
    n1m_ = sub.global_n(0); n2m_ = sub.global_n(1); n3m_ = sub.global_n(2);
    Nxh_ = n1m_ / 2 + 1;
    if (nz_loc_ % np1_ != 0) throw std::runtime_error("[PressureSolverGPU] nz_loc not divisible by np1");
    n3_I_ = nz_loc_ / np1_;
    h1p_Y_.resize(np2_); ix_start_Y_.resize(np2_);
    int base = Nxh_ / np2_, extra = Nxh_ % np2_;
    for (int r = 0; r < np2_; ++r) {
        h1p_Y_[r] = base + (r < extra ? 1 : 0);
        ix_start_Y_[r] = r * base + std::min(r, extra);
    }
    int rank_y = topo.rank_in(1);
    h1p_Y_me_ = h1p_Y_[rank_y];
    ix_start_Y_me_ = ix_start_Y_[rank_y];
    n_sys_Y_ = h1p_Y_me_ * n2m_;
    rank_xz_ = topo.rank_xz();
    size_xz_ = topo.size_xz();

    copy_vec(dx_, grid.dx(0)); copy_vec(dy_, grid.dx(1)); copy_vec(dz_, grid.dx(2));
    copy_vec(dmx_, grid.dmx(0)); copy_vec(dmy_, grid.dmx(1)); copy_vec(dmz_, grid.dmx(2));

    std::vector<double> dzg(n3m_ + 2, 0.0), dmzg(n3m_ + 2, 0.0);
    MPI_Allgather(grid.dx(2).data()+1, nz_loc_, MPI_DOUBLE, dzg.data()+1, nz_loc_, MPI_DOUBLE, topo.comm(2));
    MPI_Allgather(grid.dmx(2).data()+1, nz_loc_, MPI_DOUBLE, dmzg.data()+1, nz_loc_, MPI_DOUBLE, topo.comm(2));
    double last = grid.dmx(2)[nz_loc_ + 1];
    MPI_Bcast(&last, 1, MPI_DOUBLE, np3_ - 1, topo.comm(2));
    dmzg[n3m_ + 1] = last;
    copy_vec(dz_g_, dzg); copy_vec(dmz_g_, dmzg);

    std::vector<double> kx2(Nxh_), ky2(n2m_);
    double dx = cfg.Lx / n1m_, dy = cfg.Ly / n2m_;
    for (int i = 0; i < Nxh_; ++i) kx2[i] = 2.0*(1.0 - std::cos(2.0*M_PI*i/n1m_))/(dx*dx);
    for (int j = 0; j < n2m_; ++j) {
        int jj = (j <= n2m_/2) ? j : j - n2m_;
        ky2[j] = 2.0*(1.0 - std::cos(2.0*M_PI*jj/n2m_))/(dy*dy);
    }
    copy_vec(kx2_, kx2); copy_vec(ky2_, ky2);

    rhs_C_.reset(static_cast<std::size_t>(nx_loc_)*ny_loc_*nz_loc_);
    rhs_I_.reset(static_cast<std::size_t>(n1m_)*ny_loc_*n3_I_);
    dp_I_.reset(rhs_I_.size());
    hat_I_.reset(static_cast<std::size_t>(Nxh_)*ny_loc_*n3_I_);
    hat_Y_.reset(static_cast<std::size_t>(h1p_Y_me_)*n2m_*n3_I_);

    if (n_sys_Y_ > 0) {
        std::size_t tsz = static_cast<std::size_t>(n3_I_) * n_sys_Y_;
        tdma_Am_.reset(tsz); tdma_Ac_.reset(tsz); tdma_Ap_.reset(tsz);
        tdma_Am2_.reset(tsz); tdma_Ac2_.reset(tsz); tdma_Ap2_.reset(tsz);
        tdma_Be_r_.reset(tsz); tdma_Be_c_.reset(tsz);
        tdma_z_ = std::make_unique<PaScaLTDMACUDA>(n_sys_Y_, rank_xz_, size_xz_, topo.comm_xz(), 128, 1);
    }

    int blkC = nx_loc_ * ny_loc_ * n3_I_;
    tx_sbuf_C_.reset(static_cast<std::size_t>(np1_) * blkC);
    tx_rbuf_C_.reset(static_cast<std::size_t>(np1_) * blkC);

    int yn3 = ny_loc_ * n3_I_;
    scnts_Y_.resize(np2_); sdsp_Y_.resize(np2_); rcnts_Y_.resize(np2_); rdsp_Y_.resize(np2_);
    int off = 0;
    for (int r = 0; r < np2_; ++r) { scnts_Y_[r] = h1p_Y_[r]*yn3*2; sdsp_Y_[r] = off; off += scnts_Y_[r]; }
    int recv_each = h1p_Y_me_ * yn3 * 2;
    for (int r = 0; r < np2_; ++r) { rcnts_Y_[r] = recv_each; rdsp_Y_[r] = r * recv_each; }
    scnts_Yb_.resize(np2_); sdsp_Yb_.resize(np2_); rcnts_Yb_.resize(np2_); rdsp_Yb_.resize(np2_);
    for (int r = 0; r < np2_; ++r) { scnts_Yb_[r] = recv_each; sdsp_Yb_[r] = r * recv_each; rcnts_Yb_[r] = h1p_Y_[r]*yn3*2; }
    off = 0;
    for (int r = 0; r < np2_; ++r) { rdsp_Yb_[r] = off; off += rcnts_Yb_[r]; }
    std::size_t ybuf = std::max<std::size_t>(off, static_cast<std::size_t>(np2_) * recv_each);
    tx_sbuf_Y_.reset(ybuf); tx_rbuf_Y_.reset(ybuf);

    int n[1] = { n1m_ };
    int x_inembed[1] = { n1m_ };
    int x_onembed[1] = { Nxh_ };
    CUFFT_CHECK(cufftPlanMany(&plan_fwd_x_, 1, n, x_inembed, 1, n1m_, x_onembed, 1, Nxh_,
                              CUFFT_D2Z, ny_loc_ * n3_I_));
    CUFFT_CHECK(cufftPlanMany(&plan_bwd_x_, 1, n, x_onembed, 1, Nxh_, x_inembed, 1, n1m_,
                              CUFFT_Z2D, ny_loc_ * n3_I_));
    if (h1p_Y_me_ > 0) {
        int ny[1] = { n2m_ };
        int y_embed[1] = { n2m_ };
        CUFFT_CHECK(cufftPlanMany(&plan_y_, 1, ny, y_embed, h1p_Y_me_, 1,
                                  y_embed, h1p_Y_me_, 1, CUFFT_Z2Z, h1p_Y_me_));
    }
}

PressureSolverGPU::~PressureSolverGPU()
{
    if (plan_fwd_x_) cufftDestroy(plan_fwd_x_);
    if (plan_bwd_x_) cufftDestroy(plan_bwd_x_);
    if (plan_y_) cufftDestroy(plan_y_);
}

void PressureSolverGPU::compute_rhs_(const DeviceField& U, const DeviceField& V,
                                     const DeviceField& W, double dt)
{
    int n = nx_loc_ * ny_loc_ * nz_loc_;
    int block = 256, grid = (n + block - 1) / block;
    k_rhs<<<grid, block>>>(U.data(), V.data(), W.data(), rhs_C_.data(),
                           dx_.data(), dy_.data(), dz_.data(), nx_loc_, ny_loc_, nz_loc_,
                           1.0 / dt);
    CHANNEL_CUDA_CHECK(cudaGetLastError());
}

void PressureSolverGPU::transpose_C_to_I_()
{
    int blk = nx_loc_ * ny_loc_ * n3_I_;
    int n = blk * np1_;
    int block = 256, grid = (n + block - 1) / block;
    k_pack_C_to_I<<<grid, block>>>(rhs_C_.data(), tx_sbuf_C_.data(), nx_loc_, ny_loc_, nz_loc_, n3_I_, np1_);
    CHANNEL_CUDA_CHECK(cudaGetLastError());
    CHANNEL_CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Alltoall(tx_sbuf_C_.data(), blk, MPI_DOUBLE, tx_rbuf_C_.data(), blk, MPI_DOUBLE, topo_.comm_x());
    k_unpack_C_to_I<<<grid, block>>>(tx_rbuf_C_.data(), rhs_I_.data(), nx_loc_, ny_loc_, n1m_, n3_I_, np1_);
    CHANNEL_CUDA_CHECK(cudaGetLastError());
}

void PressureSolverGPU::transpose_I_to_C_()
{
    int blk = nx_loc_ * ny_loc_ * n3_I_;
    int n = blk * np1_;
    int block = 256, grid = (n + block - 1) / block;
    k_pack_I_to_C<<<grid, block>>>(dp_I_.data(), tx_sbuf_C_.data(), nx_loc_, ny_loc_, n1m_, n3_I_, np1_);
    CHANNEL_CUDA_CHECK(cudaGetLastError());
    CHANNEL_CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Alltoall(tx_sbuf_C_.data(), blk, MPI_DOUBLE, tx_rbuf_C_.data(), blk, MPI_DOUBLE, topo_.comm_x());
    dp_.fill(0.0);
    k_unpack_I_to_C<<<grid, block>>>(tx_rbuf_C_.data(), dp_.data(), nx_loc_, ny_loc_, nz_loc_, n3_I_, np1_);
    CHANNEL_CUDA_CHECK(cudaGetLastError());
}

void PressureSolverGPU::transpose_I_to_Y_()
{
    int yn3 = ny_loc_ * n3_I_;
    int block = 256;
    for (int r = 0; r < np2_; ++r) {
        int n = h1p_Y_[r] * yn3;
        if (n == 0) continue;
        k_pack_I_to_Y<<<(n+block-1)/block, block>>>(hat_I_.data(), tx_sbuf_Y_.data(),
                                                    r, ix_start_Y_[r], h1p_Y_[r], sdsp_Y_[r],
                                                    Nxh_, ny_loc_, n3_I_);
    }
    CHANNEL_CUDA_CHECK(cudaGetLastError());
    CHANNEL_CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Alltoallv(tx_sbuf_Y_.data(), scnts_Y_.data(), sdsp_Y_.data(), MPI_DOUBLE,
                  tx_rbuf_Y_.data(), rcnts_Y_.data(), rdsp_Y_.data(), MPI_DOUBLE,
                  topo_.comm_y());
    for (int r = 0; r < np2_; ++r) {
        int n = h1p_Y_me_ * yn3;
        if (n == 0) continue;
        k_unpack_I_to_Y<<<(n+block-1)/block, block>>>(tx_rbuf_Y_.data(), hat_Y_.data(),
                                                      r, rdsp_Y_[r], h1p_Y_me_,
                                                      ny_loc_, n2m_, n3_I_);
    }
    CHANNEL_CUDA_CHECK(cudaGetLastError());
}

void PressureSolverGPU::transpose_Y_to_I_()
{
    int yn3 = ny_loc_ * n3_I_;
    int block = 256;
    for (int r = 0; r < np2_; ++r) {
        int n = h1p_Y_me_ * yn3;
        if (n == 0) continue;
        k_pack_Y_to_I<<<(n+block-1)/block, block>>>(hat_Y_.data(), tx_sbuf_Y_.data(),
                                                    r, sdsp_Yb_[r], h1p_Y_me_,
                                                    ny_loc_, n2m_, n3_I_);
    }
    CHANNEL_CUDA_CHECK(cudaGetLastError());
    CHANNEL_CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Alltoallv(tx_sbuf_Y_.data(), scnts_Yb_.data(), sdsp_Yb_.data(), MPI_DOUBLE,
                  tx_rbuf_Y_.data(), rcnts_Yb_.data(), rdsp_Yb_.data(), MPI_DOUBLE,
                  topo_.comm_y());
    for (int r = 0; r < np2_; ++r) {
        int n = h1p_Y_[r] * yn3;
        if (n == 0) continue;
        k_unpack_Y_to_I<<<(n+block-1)/block, block>>>(tx_rbuf_Y_.data(), hat_I_.data(),
                                                      r, ix_start_Y_[r], h1p_Y_[r], rdsp_Yb_[r],
                                                      Nxh_, ny_loc_, n3_I_);
    }
    CHANNEL_CUDA_CHECK(cudaGetLastError());
}

void PressureSolverGPU::solve_tdma_z_()
{
    if (!tdma_z_ || n_sys_Y_ == 0) return;
    int n = n3_I_ * n_sys_Y_;
    int block = 256, grid = (n + block - 1) / block;
    k_build_poisson_tdma<<<grid, block>>>(hat_Y_.data(), tdma_Am_.data(), tdma_Ac_.data(), tdma_Ap_.data(),
                                          tdma_Am2_.data(), tdma_Ac2_.data(), tdma_Ap2_.data(),
                                          tdma_Be_r_.data(), tdma_Be_c_.data(),
                                          dz_g_.data(), dmz_g_.data(), kx2_.data(), ky2_.data(),
                                          n3_I_, n_sys_Y_, h1p_Y_me_, n2m_, ix_start_Y_me_,
                                          rank_xz_, size_xz_);
    CHANNEL_CUDA_CHECK(cudaGetLastError());
    if (rank_xz_ == 0 && ix_start_Y_me_ == 0 && h1p_Y_me_ > 0) {
        k_zero_mode<<<1,1>>>(tdma_Am_.data(), tdma_Ac_.data(), tdma_Ap_.data(),
                             tdma_Am2_.data(), tdma_Ac2_.data(), tdma_Ap2_.data(),
                             tdma_Be_r_.data(), tdma_Be_c_.data());
        CHANNEL_CUDA_CHECK(cudaGetLastError());
    }
    tdma_z_->solve(tdma_Am_.data(), tdma_Ac_.data(), tdma_Ap_.data(), tdma_Be_r_.data(), n_sys_Y_, n3_I_);
    tdma_z_->solve(tdma_Am2_.data(), tdma_Ac2_.data(), tdma_Ap2_.data(), tdma_Be_c_.data(), n_sys_Y_, n3_I_);
    k_write_poisson_solution<<<grid, block>>>(hat_Y_.data(), tdma_Be_r_.data(), tdma_Be_c_.data(),
                                              n3_I_, n_sys_Y_, h1p_Y_me_, n2m_);
    CHANNEL_CUDA_CHECK(cudaGetLastError());
}

void PressureSolverGPU::debug_poisson_residual_() const
{
    if (std::getenv("CHANNEL_GPU_DEBUG") == nullptr) return;

    constexpr int nblocks = 256;
    constexpr int block = 256;
    DeviceBuffer<double> part_rhs(nblocks), part_lap(nblocks), part_res(nblocks);
    k_poisson_residual_blocks<<<nblocks, block, 3 * block * sizeof(double)>>>(
        dp_.data(), rhs_C_.data(), dx_.data(), dy_.data(), dz_.data(),
        dmx_.data(), dmy_.data(), dmz_.data(),
        part_rhs.data(), part_lap.data(), part_res.data(),
        nx_loc_, ny_loc_, nz_loc_);
    CHANNEL_CUDA_CHECK(cudaGetLastError());
    CHANNEL_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> h_rhs(nblocks), h_lap(nblocks), h_res(nblocks);
    CHANNEL_CUDA_CHECK(cudaMemcpy(h_rhs.data(), part_rhs.data(), nblocks * sizeof(double), cudaMemcpyDeviceToHost));
    CHANNEL_CUDA_CHECK(cudaMemcpy(h_lap.data(), part_lap.data(), nblocks * sizeof(double), cudaMemcpyDeviceToHost));
    CHANNEL_CUDA_CHECK(cudaMemcpy(h_res.data(), part_res.data(), nblocks * sizeof(double), cudaMemcpyDeviceToHost));
    double local[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < nblocks; ++i) {
        local[0] = std::max(local[0], h_rhs[i]);
        local[1] = std::max(local[1], h_lap[i]);
        local[2] = std::max(local[2], h_res[i]);
    }
    double global[3] = {0.0, 0.0, 0.0};
    MPI_Allreduce(local, global, 3, MPI_DOUBLE, MPI_MAX, topo_.cart());
    if (topo_.rank() == 0) {
        std::printf("[dbg] poisson rhs=%.9e lap(dp)=%.9e residual=%.9e\n",
                    global[0], global[1], global[2]);
    }
}

void PressureSolverGPU::project_(DeviceField& U, DeviceField& V, DeviceField& W,
                                 DeviceField& P, double dt)
{
    int block = 256;
    int nxy = nx_loc_ * ny_loc_;
    k_dp_wall<<<(nxy+block-1)/block, block>>>(dp_.data(), nx_loc_, ny_loc_, nz_loc_,
                                              topo_.coord(2) == 0 ? 1 : 0,
                                              topo_.coord(2) == topo_.dim(2)-1 ? 1 : 0);
    CHANNEL_CUDA_CHECK(cudaGetLastError());
    halo_.exchange(dp_);
    debug_poisson_residual_();
    int n = nx_loc_ * ny_loc_ * nz_loc_;
    k_project<<<(n+block-1)/block, block>>>(U.data(), V.data(), W.data(), P.data(), dp_.data(),
                                            dmx_.data(), dmy_.data(), dmz_.data(),
                                            nx_loc_, ny_loc_, nz_loc_, dt);
    CHANNEL_CUDA_CHECK(cudaGetLastError());
}

void PressureSolverGPU::solve(DeviceField& U, DeviceField& V, DeviceField& W,
                              DeviceField& P, double dt)
{
    compute_rhs_(U, V, W, dt);
    transpose_C_to_I_();
    CUFFT_CHECK(cufftExecD2Z(plan_fwd_x_, rhs_I_.data(), hat_I_.data()));
    transpose_I_to_Y_();
    if (plan_y_) {
        std::size_t stride = static_cast<std::size_t>(h1p_Y_me_) * n2m_;
        for (int k = 0; k < n3_I_; ++k)
            CUFFT_CHECK(cufftExecZ2Z(plan_y_, hat_Y_.data() + k*stride,
                                     hat_Y_.data() + k*stride, CUFFT_FORWARD));
    }
    solve_tdma_z_();
    if (plan_y_) {
        std::size_t stride = static_cast<std::size_t>(h1p_Y_me_) * n2m_;
        for (int k = 0; k < n3_I_; ++k)
            CUFFT_CHECK(cufftExecZ2Z(plan_y_, hat_Y_.data() + k*stride,
                                     hat_Y_.data() + k*stride, CUFFT_INVERSE));
    }
    transpose_Y_to_I_();
    CUFFT_CHECK(cufftExecZ2D(plan_bwd_x_, hat_I_.data(), dp_I_.data()));
    int block = 256;
    int grid = static_cast<int>((dp_I_.size() + block - 1) / block);
    if (grid > 65535) grid = 65535;
    k_scale<<<grid, block>>>(dp_I_.data(), dp_I_.size(), 1.0 / (static_cast<double>(n1m_) * n2m_));
    CHANNEL_CUDA_CHECK(cudaGetLastError());
    transpose_I_to_C_();
    project_(U, V, W, P, dt);
}

} // namespace channel
