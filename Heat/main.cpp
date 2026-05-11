#include <mpi.h>
#include <iostream>
#include <cmath>
#include <vector>

#include "global.hpp"
#include "mpi_topology.hpp"
#include "mpi_subdomain.hpp"
#include "solve_theta.hpp"
#include "index.hpp"

int main(int argc, char** argv) {

    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // 1) Load parameters
    GlobalParams params;
    params.load(argv[1]);

    // 2) Create 3D Cartesian topology
    MPITopology topo;
    // Dirichlet walls in all three directions (matches reference setup).
    topo.init({params.np_dim[0], params.np_dim[1], params.np_dim[2]},
              {false, false, false});
    topo.make();

    auto cx = topo.commX();
    auto cy = topo.commY();
    auto cz = topo.commZ();

    int npx = params.np_dim[0], rankx = cx.myrank;
    int npy = params.np_dim[1], ranky = cy.myrank;
    int npz = params.np_dim[2], rankz = cz.myrank;

    // 3) Setup subdomain
    MPISubdomain sub;
    sub.make(params, npx, rankx, npy, ranky, npz, rankz);
    sub.makeGhostcellDDType();
    sub.mesh(params, rankx, ranky, rankz, npx, npy, npz);
    sub.indices(params, rankx, npx, ranky, npy, rankz, npz);

    // 4) Initialize theta field
    std::vector<double> theta((sub.nz_sub+1) * (sub.ny_sub+1) * (sub.nx_sub+1), 0.0);
    sub.initialization(theta);
    MPI_Barrier(MPI_COMM_WORLD);
    sub.ghostcellUpdate(theta, cx, cy, cz, params);
    sub.boundary(theta);

    // 5) Solve
    SolveTheta solver(params, topo, sub);
    solver.run(theta);
    // solver.profile(theta);

    // 6) Compute L2 error
    double local_error = 0.0;
    for (int k = 1; k < sub.nz_sub; ++k)
        for (int j = 1; j < sub.ny_sub; ++j)
            for (int i = 1; i < sub.nx_sub; ++i) {
                int ijk = idx_ijk(i, j, k, sub.nx_sub+1, sub.ny_sub+1);
                double exact = sin(Pi * sub.x_sub[i]) * sin(Pi * sub.y_sub[j])
                             * sin(Pi * sub.z_sub[k])
                             * exp(-3.0 * Pi * Pi * params.Nt * params.dt)
                             + cos(Pi * sub.x_sub[i]) * cos(Pi * sub.y_sub[j])
                             * cos(Pi * sub.z_sub[k]);
                double diff = theta[ijk] - exact;
                local_error += diff * diff;
            }

    double global_error;
    MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (myrank == 0) {
        std::cout << "Global L2 error = "
                  << std::sqrt(global_error / params.nx / params.ny / params.nz)
                  << "\n";
    }

    // 7) Cleanup
    sub.clean();
    topo.clean();
    MPI_Finalize();
    return 0;
}
