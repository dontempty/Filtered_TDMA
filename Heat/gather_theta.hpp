#ifndef GATHER_THETA_HPP
#define GATHER_THETA_HPP

#include <mpi.h>
#include <vector>
#include "mpi_subdomain.hpp"
#include "global.hpp"
#include "para_range.hpp"

/// Gather all per-rank theta fields to rank 0 into global_theta.
inline void gather_theta(std::vector<double>& global_theta,
                         std::vector<double>& theta,
                         const GlobalParams& params,
                         MPISubdomain& sub, int myrank)
{
    if (myrank == 0) {
        // Copy local data (rank 0)
        for (int k = 1; k < sub.nz_sub; ++k)
            for (int j = 1; j < sub.ny_sub; ++j)
                for (int i = 0; i < sub.nx_sub; ++i) {
                    int gk = (sub.ksta - 1) + k;
                    int gj = (sub.jsta - 1) + j;
                    int gi = (sub.ista - 1) + i;
                    int gidx = gk * (params.nx+1) * (params.ny+1) + gj * (params.nx+1) + gi;
                    int lidx = k * (sub.nx_sub+1) * (sub.ny_sub+1) + j * (sub.nx_sub+1) + i;
                    global_theta[gidx] = theta[lidx];
                }

        int Px = params.np_dim[0], Py = params.np_dim[1], Pz = params.np_dim[2];
        for (int r = 1; r < Px * Py * Pz; ++r) {
            int rkz = r % Pz;
            int rky = (r / Pz) % Py;
            int rkx = r / (Pz * Py);
            int is, ie, js, je, ks, ke;
            para_range(1, params.nx-1, Px, rkx, is, ie);
            para_range(1, params.ny-1, Py, rky, js, je);
            para_range(1, params.nz-1, Pz, rkz, ks, ke);
            int nxs = ie-is+2, nys = je-js+2, nzs = ke-ks+2;

            std::vector<int> gsizes = {params.nz+1, params.ny+1, params.nx+1};
            std::vector<int> ssizes = {nzs-1, nys-1, nxs-1};
            std::vector<int> starts = {ks, js, is};

            MPI_Datatype recvtype;
            MPI_Type_create_subarray(3, gsizes.data(), ssizes.data(), starts.data(),
                                    MPI_ORDER_C, MPI_DOUBLE, &recvtype);
            MPI_Type_commit(&recvtype);
            MPI_Recv(global_theta.data(), 1, recvtype, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Type_free(&recvtype);
        }
    } else {
        std::vector<int> gsizes = {sub.nz_sub+1, sub.ny_sub+1, sub.nx_sub+1};
        std::vector<int> ssizes = {sub.nz_sub-1, sub.ny_sub-1, sub.nx_sub-1};
        std::vector<int> starts = {1, 1, 1};

        MPI_Datatype sendtype;
        MPI_Type_create_subarray(3, gsizes.data(), ssizes.data(), starts.data(),
                                MPI_ORDER_C, MPI_DOUBLE, &sendtype);
        MPI_Type_commit(&sendtype);
        MPI_Send(theta.data(), 1, sendtype, 0, 0, MPI_COMM_WORLD);
        MPI_Type_free(&sendtype);
    }
}

#endif // GATHER_THETA_HPP
