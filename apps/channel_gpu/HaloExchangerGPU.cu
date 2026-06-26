#include "HaloExchangerGPU.hpp"
#include "MpiTopology.hpp"
#include "Subdomain.hpp"

namespace channel {

namespace gpu_kernels {

__global__ void k_pack_face(const double* f, double* lo, double* hi,
                            int nx, int ny, int nz, int axis)
{
    int nxt = nx + 2, nyt = ny + 2;
    int n = (axis == 0) ? ny * nz : ((axis == 1) ? nx * nz : nx * ny);
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;

    int i = 1, j = 1, k = 1;
    if (axis == 0) {
        j = p % ny + 1;
        k = p / ny + 1;
        lo[p] = f[df_idx(1,  j, k, nxt, nyt)];
        hi[p] = f[df_idx(nx, j, k, nxt, nyt)];
    } else if (axis == 1) {
        i = p % nx + 1;
        k = p / nx + 1;
        lo[p] = f[df_idx(i, 1,  k, nxt, nyt)];
        hi[p] = f[df_idx(i, ny, k, nxt, nyt)];
    } else {
        i = p % nx + 1;
        j = p / nx + 1;
        lo[p] = f[df_idx(i, j, 1,  nxt, nyt)];
        hi[p] = f[df_idx(i, j, nz, nxt, nyt)];
    }
}

__global__ void k_unpack_face(double* f, const double* lo, const double* hi,
                              int nx, int ny, int nz, int axis,
                              int have_lo, int have_hi)
{
    int nxt = nx + 2, nyt = ny + 2;
    int n = (axis == 0) ? ny * nz : ((axis == 1) ? nx * nz : nx * ny);
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;

    int i = 1, j = 1, k = 1;
    if (axis == 0) {
        j = p % ny + 1; k = p / ny + 1;
        if (have_lo != 0) f[df_idx(0,    j, k, nxt, nyt)] = lo[p];
        if (have_hi != 0) f[df_idx(nx+1, j, k, nxt, nyt)] = hi[p];
    } else if (axis == 1) {
        i = p % nx + 1; k = p / nx + 1;
        if (have_lo != 0) f[df_idx(i, 0,    k, nxt, nyt)] = lo[p];
        if (have_hi != 0) f[df_idx(i, ny+1, k, nxt, nyt)] = hi[p];
    } else {
        i = p % nx + 1; j = p / nx + 1;
        if (have_lo != 0) f[df_idx(i, j, 0,    nxt, nyt)] = lo[p];
        if (have_hi != 0) f[df_idx(i, j, nz+1, nxt, nyt)] = hi[p];
    }
}

} // namespace gpu_kernels
using namespace gpu_kernels;

HaloExchangerGPU::HaloExchangerGPU(const MpiTopology& topo, const Subdomain& sub)
    : topo_(topo), nx_(sub.nx()), ny_(sub.ny()), nz_(sub.nz())
{
    std::size_t nface[3] = {
        static_cast<std::size_t>(ny_) * nz_,
        static_cast<std::size_t>(nx_) * nz_,
        static_cast<std::size_t>(nx_) * ny_
    };
    for (int a = 0; a < 3; ++a) {
        send_lo_[a].reset(nface[a]); send_hi_[a].reset(nface[a]);
        recv_lo_[a].reset(nface[a]); recv_hi_[a].reset(nface[a]);
    }
}

void HaloExchangerGPU::exchange_axis(DeviceField& f, int axis) const
{
    const int left = topo_.left_in(axis);
    const int right = topo_.right_in(axis);
    const int n = (axis == 0) ? ny_ * nz_ : ((axis == 1) ? nx_ * nz_ : nx_ * ny_);
    const int block = 256, grid = (n + block - 1) / block;

    k_pack_face<<<grid, block>>>(f.data(), send_lo_[axis].data(), send_hi_[axis].data(),
                                 nx_, ny_, nz_, axis);
    CHANNEL_CUDA_CHECK(cudaGetLastError());
    CHANNEL_CUDA_CHECK(cudaDeviceSynchronize());

    constexpr int TAG_LO = 100;
    constexpr int TAG_HI = 200;
    MPI_Request req[4];
    int nreq = 0;

    if (right != MPI_PROC_NULL)
        MPI_Irecv(recv_hi_[axis].data(), n, MPI_DOUBLE, right, TAG_LO, topo_.cart(), &req[nreq++]);
    if (left != MPI_PROC_NULL)
        MPI_Irecv(recv_lo_[axis].data(), n, MPI_DOUBLE, left, TAG_HI, topo_.cart(), &req[nreq++]);
    if (left != MPI_PROC_NULL)
        MPI_Isend(send_lo_[axis].data(), n, MPI_DOUBLE, left, TAG_LO, topo_.cart(), &req[nreq++]);
    if (right != MPI_PROC_NULL)
        MPI_Isend(send_hi_[axis].data(), n, MPI_DOUBLE, right, TAG_HI, topo_.cart(), &req[nreq++]);
    if (nreq > 0) MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);

    k_unpack_face<<<grid, block>>>(f.data(), recv_lo_[axis].data(), recv_hi_[axis].data(),
                                   nx_, ny_, nz_, axis,
                                   left != MPI_PROC_NULL ? 1 : 0,
                                   right != MPI_PROC_NULL ? 1 : 0);
    CHANNEL_CUDA_CHECK(cudaGetLastError());
}

void HaloExchangerGPU::exchange(DeviceField& f) const
{
    exchange_axis(f, 0);
    exchange_axis(f, 1);
    exchange_axis(f, 2);
}

} // namespace channel
