#include "BoundaryConditionGPU.hpp"
#include "MpiTopology.hpp"
#include "Subdomain.hpp"

__global__ void k_wall_bc(double* U, double* V, double* W,
                          int nx, int ny, int nz, int low, int high)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int n = (nx + 2) * (ny + 2);
    if (p >= n) return;
    int i = p % (nx + 2);
    int j = p / (nx + 2);
    int nxt = nx + 2, nyt = ny + 2;
    if (low != 0) {
        U[channel::df_idx(i,j,0,nxt,nyt)] = -U[channel::df_idx(i,j,1,nxt,nyt)];
        V[channel::df_idx(i,j,0,nxt,nyt)] = -V[channel::df_idx(i,j,1,nxt,nyt)];
        W[channel::df_idx(i,j,1,nxt,nyt)] = 0.0;
    }
    if (high != 0) {
        U[channel::df_idx(i,j,nz+1,nxt,nyt)] = -U[channel::df_idx(i,j,nz,nxt,nyt)];
        V[channel::df_idx(i,j,nz+1,nxt,nyt)] = -V[channel::df_idx(i,j,nz,nxt,nyt)];
        W[channel::df_idx(i,j,nz+1,nxt,nyt)] = 0.0;
    }
}

namespace channel {

BoundaryConditionGPU::BoundaryConditionGPU(const MpiTopology& topo, const Subdomain& sub)
    : nx_(sub.nx()), ny_(sub.ny()), nz_(sub.nz())
{
    if (!topo.periodic(2)) {
        low_ = (topo.rank_in(2) == 0);
        high_ = (topo.rank_in(2) == topo.size_in(2) - 1);
    }
}

void BoundaryConditionGPU::apply(DeviceField& U, DeviceField& V, DeviceField& W) const
{
    int n = (nx_ + 2) * (ny_ + 2);
    int block = 256, grid = (n + block - 1) / block;
    k_wall_bc<<<grid, block>>>(U.data(), V.data(), W.data(), nx_, ny_, nz_,
                               low_ ? 1 : 0, high_ ? 1 : 0);
    CHANNEL_CUDA_CHECK(cudaGetLastError());
}

} // namespace channel
