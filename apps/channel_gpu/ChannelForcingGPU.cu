#include "ChannelForcingGPU.hpp"
#include "Config.hpp"
#include "Grid.hpp"
#include "MpiTopology.hpp"
#include "Subdomain.hpp"

#include <mpi.h>
#include <vector>

namespace channel {

namespace gpu_kernels {

__global__ void k_bulk_blocks(const double* U, const double* dx, const double* dy, const double* dz,
                              double* part, int nx, int ny, int nz)
{
    extern __shared__ double sh[];
    int tid = threadIdx.x;
    int n = nx * ny * nz;
    double sum = 0.0;
    int nxt = nx + 2, nyt = ny + 2;
    for (int p = blockIdx.x * blockDim.x + tid; p < n; p += blockDim.x * gridDim.x) {
        int i = p % nx + 1, j = (p / nx) % ny + 1, k = p / (nx * ny) + 1;
        double dV = dx[i] * dy[j] * dz[k];
        sum += 0.5 * (U[df_idx(i,j,k,nxt,nyt)] + U[df_idx(i+1,j,k,nxt,nyt)]) * dV;
    }
    sh[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid+s];
        __syncthreads();
    }
    if (tid == 0) part[blockIdx.x] = sh[0];
}

__global__ void k_reduce_blocks(const double* part, double* out, int n)
{
    extern __shared__ double sh[];
    int tid = threadIdx.x;
    double sum = 0.0;
    for (int i = tid; i < n; i += blockDim.x) sum += part[i];
    sh[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid+s];
        __syncthreads();
    }
    if (tid == 0) out[0] = sh[0];
}

__global__ void k_set_scalar(double* p, double v) { p[0] = v; }

__global__ void k_mass_correct(double* U, double* dPdx, const double* bulk_sum,
                               int nx, int ny, int nz, double total_volume,
                               double target_bulk, double dt)
{
    double Ub = bulk_sum[0] / total_volume;
    double DMpresg = (dt > 1.0e-15) ? (Ub - target_bulk) / dt : 0.0;
    double shift = -dt * DMpresg;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nx * ny * nz;
    int nxt = nx + 2, nyt = ny + 2;
    for (int q = p; q < n; q += blockDim.x * gridDim.x) {
        int i = q % nx + 1, j = (q / nx) % ny + 1, k = q / (nx * ny) + 1;
        U[df_idx(i,j,k,nxt,nyt)] += shift;
    }
    if (p == 0) dPdx[0] += DMpresg;
}

} // namespace gpu_kernels
using namespace gpu_kernels;

static void copy_vec(DeviceBuffer<double>& d, const std::vector<double>& h)
{
    d.reset(h.size());
    CHANNEL_CUDA_CHECK(cudaMemcpy(d.data(), h.data(), h.size()*sizeof(double), cudaMemcpyHostToDevice));
}

ChannelForcingGPU::ChannelForcingGPU(const Config& cfg, const MpiTopology& topo,
                                     const Subdomain& sub, const Grid& grid)
    : cfg_(cfg), topo_(topo), nx_(sub.nx()), ny_(sub.ny()), nz_(sub.nz()),
      total_volume_(cfg.Lx * cfg.Ly * cfg.Lz)
{
    copy_vec(dx_, grid.dx(0)); copy_vec(dy_, grid.dx(1)); copy_vec(dz_, grid.dx(2));
    d_part_.reset(256);
    d_sum_.reset(1);
    d_global_sum_.reset(1);
    d_dPdx_.reset(1);
    set_mean_dPdx(cfg.target_dPdx);
}

void ChannelForcingGPU::set_mean_dPdx(double v)
{
    k_set_scalar<<<1,1>>>(d_dPdx_.data(), v);
    CHANNEL_CUDA_CHECK(cudaGetLastError());
}

double ChannelForcingGPU::mean_dPdx_host() const
{
    double v = 0.0;
    CHANNEL_CUDA_CHECK(cudaMemcpy(&v, d_dPdx_.data(), sizeof(double), cudaMemcpyDeviceToHost));
    return v;
}

double ChannelForcingGPU::bulk_velocity_host(const DeviceField& U) const
{
    k_bulk_blocks<<<256,256,256*sizeof(double)>>>(U.data(), dx_.data(), dy_.data(), dz_.data(),
                                                  d_part_.data(), nx_, ny_, nz_);
    k_reduce_blocks<<<1,256,256*sizeof(double)>>>(d_part_.data(), d_sum_.data(), 256);
    CHANNEL_CUDA_CHECK(cudaGetLastError());
    CHANNEL_CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Allreduce(d_sum_.data(), d_global_sum_.data(), 1, MPI_DOUBLE, MPI_SUM, topo_.cart());
    double s = 0.0;
    CHANNEL_CUDA_CHECK(cudaMemcpy(&s, d_global_sum_.data(), sizeof(double), cudaMemcpyDeviceToHost));
    return s / total_volume_;
}

void ChannelForcingGPU::correct(DeviceField& U, double dt)
{
    k_bulk_blocks<<<256,256,256*sizeof(double)>>>(U.data(), dx_.data(), dy_.data(), dz_.data(),
                                                  d_part_.data(), nx_, ny_, nz_);
    k_reduce_blocks<<<1,256,256*sizeof(double)>>>(d_part_.data(), d_sum_.data(), 256);
    CHANNEL_CUDA_CHECK(cudaGetLastError());
    CHANNEL_CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Allreduce(d_sum_.data(), d_global_sum_.data(), 1, MPI_DOUBLE, MPI_SUM, topo_.cart());
    int n = nx_ * ny_ * nz_;
    k_mass_correct<<<256,256>>>(U.data(), d_dPdx_.data(), d_global_sum_.data(),
                                nx_, ny_, nz_, total_volume_, cfg_.target_bulk_velocity, dt);
    CHANNEL_CUDA_CHECK(cudaGetLastError());
}

} // namespace channel
