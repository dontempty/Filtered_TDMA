#include "StatisticsGPU.hpp"
#include "Grid.hpp"
#include "MpiTopology.hpp"
#include "Subdomain.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <mpi.h>

namespace channel {

namespace gpu_kernels {

__global__ void k_zero(double* a, int n)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < n) a[p] = 0.0;
}

__global__ void k_acc_stats(const double* U, const double* V, const double* W, const double* P,
                            double* Um, double* U2m, double* Vm, double* V2m,
                            double* Wcm, double* Wc2m, double* Pm,
                            double* Ugm, double* Wgm, double* UWgm,
                            const double* dx, const double* dy, const double* dz,
                            int nx, int ny, int nz, int nxg, int nyg, double inv_n)
{
    int kl = blockIdx.x + 1;
    int tid = threadIdx.x;
    extern __shared__ double sh[];
    double* su = sh;       double* su2 = su + blockDim.x;
    double* sv = su2 + blockDim.x; double* sv2 = sv + blockDim.x;
    double* sw = sv2 + blockDim.x; double* sw2 = sw + blockDim.x;
    double* sp = sw2 + blockDim.x; double* sug = sp + blockDim.x;
    double* swg = sug + blockDim.x; double* suwg = swg + blockDim.x;
    double a=0,b=0,c=0,d=0,e=0,f=0,g=0,h=0,l=0,m=0;
    int nxt = nx + 2, nyt = ny + 2;
    for (int q = tid; q < nx*ny; q += blockDim.x) {
        int i = q % nx + 1, j = q / nx + 1;
        double uc = 0.5 * (U[df_idx(i,j,kl,nxt,nyt)] + U[df_idx(i+1,j,kl,nxt,nyt)]);
        double vc = 0.5 * (V[df_idx(i,j,kl,nxt,nyt)] + V[df_idx(i,j+1,kl,nxt,nyt)]);
        double wc = 0.5 * (W[df_idx(i,j,kl,nxt,nyt)] + W[df_idx(i,j,kl+1,nxt,nyt)]);
        double pp = P[df_idx(i,j,kl,nxt,nyt)];
        a += uc; b += uc*uc; c += vc; d += vc*vc; e += wc; f += wc*wc; g += pp;

        int im = i - 1, jm = j - 1, km = kl - 1;
        double zsum = dz[kl] + dz[km];
        double ysum = dy[j] + dy[jm];
        double xsum = dx[i] + dx[im];
        double Uy_km = (U[df_idx(i,j,km,nxt,nyt)]*dy[jm] + U[df_idx(i,jm,km,nxt,nyt)]*dy[j]) / ysum;
        double Uy_kl = (U[df_idx(i,j,kl,nxt,nyt)]*dy[jm] + U[df_idx(i,jm,kl,nxt,nyt)]*dy[j]) / ysum;
        double Ug = (Uy_kl*dz[km] + Uy_km*dz[kl]) / zsum;
        double Wx_jm = (W[df_idx(i,jm,kl,nxt,nyt)]*dx[im] + W[df_idx(im,jm,kl,nxt,nyt)]*dx[i]) / xsum;
        double Wx_j  = (W[df_idx(i,j, kl,nxt,nyt)]*dx[im] + W[df_idx(im,j, kl,nxt,nyt)]*dx[i]) / xsum;
        double Wg = (Wx_j*dy[jm] + Wx_jm*dy[j]) / ysum;
        h += Ug; l += Wg; m += Ug * Wg;
    }
    su[tid]=a; su2[tid]=b; sv[tid]=c; sv2[tid]=d; sw[tid]=e; sw2[tid]=f;
    sp[tid]=g; sug[tid]=h; swg[tid]=l; suwg[tid]=m;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            su[tid]+=su[tid+s]; su2[tid]+=su2[tid+s]; sv[tid]+=sv[tid+s]; sv2[tid]+=sv2[tid+s];
            sw[tid]+=sw[tid+s]; sw2[tid]+=sw2[tid+s]; sp[tid]+=sp[tid+s]; sug[tid]+=sug[tid+s];
            swg[tid]+=swg[tid+s]; suwg[tid]+=suwg[tid+s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        int k0 = kl - 1;
        double inv = 1.0 / static_cast<double>(nxg * nyg);
        double vals[10] = {su[0]*inv, su2[0]*inv, sv[0]*inv, sv2[0]*inv, sw[0]*inv,
                           sw2[0]*inv, sp[0]*inv, sug[0]*inv, swg[0]*inv, suwg[0]*inv};
        Um[k0]   += (vals[0] - Um[k0])   * inv_n;
        U2m[k0]  += (vals[1] - U2m[k0])  * inv_n;
        Vm[k0]   += (vals[2] - Vm[k0])   * inv_n;
        V2m[k0]  += (vals[3] - V2m[k0])  * inv_n;
        Wcm[k0]  += (vals[4] - Wcm[k0])  * inv_n;
        Wc2m[k0] += (vals[5] - Wc2m[k0]) * inv_n;
        Pm[k0]   += (vals[6] - Pm[k0])   * inv_n;
        Ugm[k0]  += (vals[7] - Ugm[k0])  * inv_n;
        Wgm[k0]  += (vals[8] - Wgm[k0])  * inv_n;
        UWgm[k0] += (vals[9] - UWgm[k0]) * inv_n;
    }
}

} // namespace gpu_kernels
using namespace gpu_kernels;

static void copy_vec(DeviceBuffer<double>& d, const std::vector<double>& h)
{
    d.reset(h.size());
    CHANNEL_CUDA_CHECK(cudaMemcpy(d.data(), h.data(), h.size()*sizeof(double), cudaMemcpyHostToDevice));
}

StatisticsGPU::StatisticsGPU(const MpiTopology& topo, const Subdomain& sub, const Grid& grid)
    : topo_(topo), nx_(sub.nx()), ny_(sub.ny()), nz_local_(sub.nz()),
      nz_global_(sub.global_n(2)), kstart_(sub.ista(2) - 1),
      nx_global_(sub.global_n(0)), ny_global_(sub.global_n(1))
{
    copy_vec(dx_, grid.dx(0)); copy_vec(dy_, grid.dx(1)); copy_vec(dz_, grid.dx(2));
    U_m_.reset(nz_local_); U2_m_.reset(nz_local_); V_m_.reset(nz_local_); V2_m_.reset(nz_local_);
    Wc_m_.reset(nz_local_); Wc2_m_.reset(nz_local_); P_m_.reset(nz_local_);
    Ug_m_.reset(nz_local_); Wg_m_.reset(nz_local_); UWg_m_.reset(nz_local_);
    reset();

    std::vector<double> dz_local(nz_local_), dz_global(nz_global_);
    for (int k = 0; k < nz_local_; ++k) dz_local[k] = grid.dx(2)[k+1];
    MPI_Allgather(dz_local.data(), nz_local_, MPI_DOUBLE,
                  dz_global.data(), nz_local_, MPI_DOUBLE, topo.comm_z());
    zc_global_.resize(nz_global_); z_face_global_.resize(nz_global_);
    double cum = 0.0;
    for (int k = 0; k < nz_global_; ++k) {
        z_face_global_[k] = cum;
        zc_global_[k] = cum + 0.5 * dz_global[k];
        cum += dz_global[k];
    }
}

void StatisticsGPU::reset()
{
    n_ = 0;
    int block = 256, grid = (nz_local_ + block - 1) / block;
    double* arrays[] = {U_m_.data(), U2_m_.data(), V_m_.data(), V2_m_.data(), Wc_m_.data(),
                        Wc2_m_.data(), P_m_.data(), Ug_m_.data(), Wg_m_.data(), UWg_m_.data()};
    for (double* p : arrays) k_zero<<<grid, block>>>(p, nz_local_);
    CHANNEL_CUDA_CHECK(cudaGetLastError());
}

void StatisticsGPU::accumulate(const DeviceField& U, const DeviceField& V,
                               const DeviceField& W, const DeviceField& P)
{
    ++n_;
    k_acc_stats<<<nz_local_, 256, 10*256*sizeof(double)>>>(
        U.data(), V.data(), W.data(), P.data(),
        U_m_.data(), U2_m_.data(), V_m_.data(), V2_m_.data(),
        Wc_m_.data(), Wc2_m_.data(), P_m_.data(),
        Ug_m_.data(), Wg_m_.data(), UWg_m_.data(),
        dx_.data(), dy_.data(), dz_.data(),
        nx_, ny_, nz_local_, nx_global_, ny_global_, 1.0 / static_cast<double>(n_));
    CHANNEL_CUDA_CHECK(cudaGetLastError());
}

static void gather_one(const std::vector<double>& loc, std::vector<double>& glob,
                       int nz_global, int kstart, MPI_Comm comm)
{
    std::vector<double> tmp(nz_global, 0.0);
    for (std::size_t i = 0; i < loc.size(); ++i) tmp[kstart + static_cast<int>(i)] = loc[i];
    glob.assign(nz_global, 0.0);
    MPI_Allreduce(tmp.data(), glob.data(), nz_global, MPI_DOUBLE, MPI_SUM, comm);
}

void StatisticsGPU::write(const std::string& path, int step, double Re_b, bool reset_after)
{
    std::vector<double> loc[10], glob[10];
    DeviceBuffer<double>* src[10] = {&U_m_, &U2_m_, &V_m_, &V2_m_, &Wc_m_, &Wc2_m_, &P_m_, &Ug_m_, &Wg_m_, &UWg_m_};
    for (int f = 0; f < 10; ++f) {
        loc[f].resize(nz_local_);
        CHANNEL_CUDA_CHECK(cudaMemcpy(loc[f].data(), src[f]->data(), nz_local_*sizeof(double), cudaMemcpyDeviceToHost));
        gather_one(loc[f], glob[f], nz_global_, kstart_, topo_.cart());
    }
    long n_saved = n_;
    if (reset_after) reset();
    if (topo_.rank() != 0 || n_saved == 0) return;

    double nu = 1.0 / Re_b;
    double tau_w = nu * std::abs(glob[0][0]) / zc_global_[0];
    double u_tau = std::sqrt(std::max(tau_w, 0.0));
    FILE* fp = std::fopen(path.c_str(), "w");
    if (!fp) return;
    std::fprintf(fp, "TITLE = \"Channel Flow Statistics (step=%d, n_samples=%ld)\"\n", step, n_saved);
    std::fprintf(fp, "VARIABLES = \"Z\" \"Z_plus\" \"U_mean\" \"W_mean\" \"u_rms\" \"v_rms\" \"w_rms\" \"uw_stress\" \"P_mean\"\n");
    std::fprintf(fp, "ZONE T=\"Stats\", I=%d, J=1, K=1, DATAPACKING=POINT\n", nz_global_);
    for (int k = 0; k < nz_global_; ++k) {
        double u_rms = std::sqrt(std::max(glob[1][k] - glob[0][k]*glob[0][k], 0.0));
        double v_rms = std::sqrt(std::max(glob[3][k] - glob[2][k]*glob[2][k], 0.0));
        double w_rms = std::sqrt(std::max(glob[5][k] - glob[4][k]*glob[4][k], 0.0));
        double uw = glob[9][k] - glob[7][k] * glob[8][k];
        if (k + 1 < nz_global_) {
            double uw2 = glob[9][k+1] - glob[7][k+1] * glob[8][k+1];
            uw = 0.5 * (uw + uw2);
        }
        std::fprintf(fp, "%.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e\n",
                     zc_global_[k], zc_global_[k]*u_tau*Re_b, glob[0][k], glob[4][k],
                     u_rms, v_rms, w_rms, uw, glob[6][k]);
    }
    std::fclose(fp);
}

} // namespace channel
