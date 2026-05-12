// Device-side ghost-cell exchange for Heat_gpu.
//
// Pattern (mirrors PaScaL_TDMA_F examples/solve_theta.f90 :: ghostcell_update_cuda):
//   1. pack the 6 face slabs of d_theta into 12 contiguous device send buffers
//      (sbuf_x0/x1, sbuf_y0/y1, sbuf_z0/z1) with simple 2D CUDA kernels.
//   2. MPI_Isend / MPI_Irecv on the CONTIGUOUS device pointers — this is the
//      fast path for CUDA-aware MPI (UCX `cuda_ipc` / `cuda_copy`).  No
//      derived datatypes on device pointers — those force OpenMPI into a
//      per-cell cudaMemcpy or D2H-staging path which on V100 PCIe nodes is
//      ~300× slower than the contiguous variant.
//   3. unpack the recv buffers back into the ghost slots of d_theta.
//
// Boundary ranks: if west_rank / east_rank == MPI_PROC_NULL the corresponding
// pack/unpack is skipped — the time-independent Dirichlet wall value placed by
// the initial H2D copy of theta_sub stays valid for the whole run.

#include "mpi_subdomain.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            std::fprintf(stderr, "[CUDA] %s:%d %s -> %s\n", __FILE__,          \
                         __LINE__, #expr, cudaGetErrorString(_err));           \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// d_theta is row-major [(nx_sub+1) × (ny_sub+1) × (nz_sub+1)] with i fastest:
//   idx(i,j,k) = k*(ny+1)*(nx+1) + j*(nx+1) + i.
static __device__ __forceinline__
std::size_t idx3(int i, int j, int k, int nx1, int ny1) {
    return ((std::size_t)k * ny1 + j) * nx1 + i;
}

// ----- Pack kernels (gather one face slab into contiguous buffer) ----------

__global__ void pack_x(double* __restrict__ sbuf, const double* __restrict__ d_theta,
                       int i_src, int nx1, int ny1, int nz1) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= ny1 || k >= nz1) return;
    sbuf[(std::size_t)k * ny1 + j] = d_theta[idx3(i_src, j, k, nx1, ny1)];
}

__global__ void pack_y(double* __restrict__ sbuf, const double* __restrict__ d_theta,
                       int j_src, int nx1, int ny1, int nz1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx1 || k >= nz1) return;
    sbuf[(std::size_t)k * nx1 + i] = d_theta[idx3(i, j_src, k, nx1, ny1)];
}

__global__ void pack_z(double* __restrict__ sbuf, const double* __restrict__ d_theta,
                       int k_src, int nx1, int ny1, int nz1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx1 || j >= ny1) return;
    sbuf[(std::size_t)j * nx1 + i] = d_theta[idx3(i, j, k_src, nx1, ny1)];
}

// ----- Unpack kernels (scatter contiguous buffer into ghost slot) ----------

__global__ void unpack_x(double* __restrict__ d_theta, const double* __restrict__ rbuf,
                         int i_dst, int nx1, int ny1, int nz1) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= ny1 || k >= nz1) return;
    d_theta[idx3(i_dst, j, k, nx1, ny1)] = rbuf[(std::size_t)k * ny1 + j];
}

__global__ void unpack_y(double* __restrict__ d_theta, const double* __restrict__ rbuf,
                         int j_dst, int nx1, int ny1, int nz1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx1 || k >= nz1) return;
    d_theta[idx3(i, j_dst, k, nx1, ny1)] = rbuf[(std::size_t)k * nx1 + i];
}

__global__ void unpack_z(double* __restrict__ d_theta, const double* __restrict__ rbuf,
                         int k_dst, int nx1, int ny1, int nz1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx1 || j >= ny1) return;
    d_theta[idx3(i, j, k_dst, nx1, ny1)] = rbuf[(std::size_t)j * nx1 + i];
}

// ----- Allocation / free ---------------------------------------------------

void MPISubdomain::allocGhostBufsDevice() {
    const int nx1 = nx_sub + 1;
    const int ny1 = ny_sub + 1;
    const int nz1 = nz_sub + 1;
    const std::size_t nx_face = (std::size_t)ny1 * nz1;   // x-direction face
    const std::size_t ny_face = (std::size_t)nx1 * nz1;
    const std::size_t nz_face = (std::size_t)nx1 * ny1;

    auto alloc = [](double** p, std::size_t n) {
        if (*p) return;
        CUDA_CHECK(cudaMalloc(p, sizeof(double) * n));
    };
    alloc(&d_sbuf_x0, nx_face);  alloc(&d_sbuf_x1, nx_face);
    alloc(&d_rbuf_x0, nx_face);  alloc(&d_rbuf_x1, nx_face);
    alloc(&d_sbuf_y0, ny_face);  alloc(&d_sbuf_y1, ny_face);
    alloc(&d_rbuf_y0, ny_face);  alloc(&d_rbuf_y1, ny_face);
    alloc(&d_sbuf_z0, nz_face);  alloc(&d_sbuf_z1, nz_face);
    alloc(&d_rbuf_z0, nz_face);  alloc(&d_rbuf_z1, nz_face);
}

void MPISubdomain::freeGhostBufsDevice() {
    auto fr = [](double** p) { if (*p) { cudaFree(*p); *p = nullptr; } };
    fr(&d_sbuf_x0); fr(&d_sbuf_x1); fr(&d_rbuf_x0); fr(&d_rbuf_x1);
    fr(&d_sbuf_y0); fr(&d_sbuf_y1); fr(&d_rbuf_y0); fr(&d_rbuf_y1);
    fr(&d_sbuf_z0); fr(&d_sbuf_z1); fr(&d_rbuf_z0); fr(&d_rbuf_z1);
}

// ----- ghostcellUpdateDevice — main entry, called every timestep ----------

void MPISubdomain::ghostcellUpdateDevice(double* d_theta,
                                         const CartComm1D& cx,
                                         const CartComm1D& cy,
                                         const CartComm1D& cz) {
    const int nx1 = nx_sub + 1;
    const int ny1 = ny_sub + 1;
    const int nz1 = nz_sub + 1;
    const int nx_face = ny1 * nz1;
    const int ny_face = nx1 * nz1;
    const int nz_face = nx1 * ny1;

    dim3 blk(16, 16);
    auto g2 = [](int a, int b, dim3 bk) {
        return dim3((a + bk.x - 1) / bk.x, (b + bk.y - 1) / bk.y);
    };

    MPI_Request req[4];

    // ============ X direction ============
    if (cx.nprocs > 1) {
        if (cx.west_rank != MPI_PROC_NULL) {
            pack_x<<<g2(ny1, nz1, blk), blk>>>(d_sbuf_x0, d_theta,
                                               /*i_src=*/1, nx1, ny1, nz1);
        }
        if (cx.east_rank != MPI_PROC_NULL) {
            pack_x<<<g2(ny1, nz1, blk), blk>>>(d_sbuf_x1, d_theta,
                                               /*i_src=*/nx_sub - 1, nx1, ny1, nz1);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        MPI_Isend(d_sbuf_x0, nx_face, MPI_DOUBLE, cx.west_rank, 111, cx.comm, &req[0]);
        MPI_Irecv(d_rbuf_x1, nx_face, MPI_DOUBLE, cx.east_rank, 111, cx.comm, &req[1]);
        MPI_Isend(d_sbuf_x1, nx_face, MPI_DOUBLE, cx.east_rank, 222, cx.comm, &req[2]);
        MPI_Irecv(d_rbuf_x0, nx_face, MPI_DOUBLE, cx.west_rank, 222, cx.comm, &req[3]);
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);

        if (cx.west_rank != MPI_PROC_NULL) {
            unpack_x<<<g2(ny1, nz1, blk), blk>>>(d_theta, d_rbuf_x0,
                                                 /*i_dst=*/0, nx1, ny1, nz1);
        }
        if (cx.east_rank != MPI_PROC_NULL) {
            unpack_x<<<g2(ny1, nz1, blk), blk>>>(d_theta, d_rbuf_x1,
                                                 /*i_dst=*/nx_sub, nx1, ny1, nz1);
        }
    }

    // ============ Y direction ============
    if (cy.nprocs > 1) {
        if (cy.west_rank != MPI_PROC_NULL) {
            pack_y<<<g2(nx1, nz1, blk), blk>>>(d_sbuf_y0, d_theta,
                                               /*j_src=*/1, nx1, ny1, nz1);
        }
        if (cy.east_rank != MPI_PROC_NULL) {
            pack_y<<<g2(nx1, nz1, blk), blk>>>(d_sbuf_y1, d_theta,
                                               /*j_src=*/ny_sub - 1, nx1, ny1, nz1);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        MPI_Isend(d_sbuf_y0, ny_face, MPI_DOUBLE, cy.west_rank, 333, cy.comm, &req[0]);
        MPI_Irecv(d_rbuf_y1, ny_face, MPI_DOUBLE, cy.east_rank, 333, cy.comm, &req[1]);
        MPI_Isend(d_sbuf_y1, ny_face, MPI_DOUBLE, cy.east_rank, 444, cy.comm, &req[2]);
        MPI_Irecv(d_rbuf_y0, ny_face, MPI_DOUBLE, cy.west_rank, 444, cy.comm, &req[3]);
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);

        if (cy.west_rank != MPI_PROC_NULL) {
            unpack_y<<<g2(nx1, nz1, blk), blk>>>(d_theta, d_rbuf_y0,
                                                 /*j_dst=*/0, nx1, ny1, nz1);
        }
        if (cy.east_rank != MPI_PROC_NULL) {
            unpack_y<<<g2(nx1, nz1, blk), blk>>>(d_theta, d_rbuf_y1,
                                                 /*j_dst=*/ny_sub, nx1, ny1, nz1);
        }
    }

    // ============ Z direction ============
    if (cz.nprocs > 1) {
        if (cz.west_rank != MPI_PROC_NULL) {
            pack_z<<<g2(nx1, ny1, blk), blk>>>(d_sbuf_z0, d_theta,
                                               /*k_src=*/1, nx1, ny1, nz1);
        }
        if (cz.east_rank != MPI_PROC_NULL) {
            pack_z<<<g2(nx1, ny1, blk), blk>>>(d_sbuf_z1, d_theta,
                                               /*k_src=*/nz_sub - 1, nx1, ny1, nz1);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        MPI_Isend(d_sbuf_z0, nz_face, MPI_DOUBLE, cz.west_rank, 555, cz.comm, &req[0]);
        MPI_Irecv(d_rbuf_z1, nz_face, MPI_DOUBLE, cz.east_rank, 555, cz.comm, &req[1]);
        MPI_Isend(d_sbuf_z1, nz_face, MPI_DOUBLE, cz.east_rank, 666, cz.comm, &req[2]);
        MPI_Irecv(d_rbuf_z0, nz_face, MPI_DOUBLE, cz.west_rank, 666, cz.comm, &req[3]);
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);

        if (cz.west_rank != MPI_PROC_NULL) {
            unpack_z<<<g2(nx1, ny1, blk), blk>>>(d_theta, d_rbuf_z0,
                                                 /*k_dst=*/0, nx1, ny1, nz1);
        }
        if (cz.east_rank != MPI_PROC_NULL) {
            unpack_z<<<g2(nx1, ny1, blk), blk>>>(d_theta, d_rbuf_z1,
                                                 /*k_dst=*/nz_sub, nx1, ny1, nz1);
        }
    }
}
