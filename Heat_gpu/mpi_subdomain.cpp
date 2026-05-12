#include "mpi_subdomain.hpp"
#include "para_range.hpp"
#include "index.hpp"
#include <cmath>
#include <algorithm>

const double Pi = 3.14159265358979323846;

MPISubdomain::MPISubdomain()
    : ddtype_sendto_x_right(MPI_DATATYPE_NULL), ddtype_recvfrom_x_left(MPI_DATATYPE_NULL),
      ddtype_sendto_x_left(MPI_DATATYPE_NULL),  ddtype_recvfrom_x_right(MPI_DATATYPE_NULL),
      ddtype_sendto_y_right(MPI_DATATYPE_NULL), ddtype_recvfrom_y_left(MPI_DATATYPE_NULL),
      ddtype_sendto_y_left(MPI_DATATYPE_NULL),  ddtype_recvfrom_y_right(MPI_DATATYPE_NULL),
      ddtype_sendto_z_right(MPI_DATATYPE_NULL), ddtype_recvfrom_z_left(MPI_DATATYPE_NULL),
      ddtype_sendto_z_left(MPI_DATATYPE_NULL),  ddtype_recvfrom_z_right(MPI_DATATYPE_NULL) {}

void MPISubdomain::make(const GlobalParams& params,
                        int npx, int rankx, int npy, int ranky, int npz, int rankz) {
    para_range(1, params.nx-1, npx, rankx, ista, iend);
    nx_sub = iend - ista + 2;
    para_range(1, params.ny-1, npy, ranky, jsta, jend);
    ny_sub = jend - jsta + 2;
    para_range(1, params.nz-1, npz, rankz, ksta, kend);
    nz_sub = kend - ksta + 2;

    x_sub.resize(nx_sub + 1);   dmx_sub.resize(nx_sub + 1);
    y_sub.resize(ny_sub + 1);   dmy_sub.resize(ny_sub + 1);
    z_sub.resize(nz_sub + 1);   dmz_sub.resize(nz_sub + 1);

    theta_x_left_sub.assign((ny_sub+1)*(nz_sub+1), 0.0);
    theta_x_right_sub.assign((ny_sub+1)*(nz_sub+1), 0.0);
    theta_y_left_sub.assign((nz_sub+1)*(nx_sub+1), 0.0);
    theta_y_right_sub.assign((nz_sub+1)*(nx_sub+1), 0.0);
    theta_z_left_sub.assign((nx_sub+1)*(ny_sub+1), 0.0);
    theta_z_right_sub.assign((nx_sub+1)*(ny_sub+1), 0.0);

    theta_x_left_index.assign(nx_sub+1, 0);
    theta_x_right_index.assign(nx_sub+1, 0);
    theta_y_left_index.assign(ny_sub+1, 0);
    theta_y_right_index.assign(ny_sub+1, 0);
    theta_z_left_index.assign(nz_sub+1, 0);
    theta_z_right_index.assign(nz_sub+1, 0);
}

void MPISubdomain::clean() {
    auto freeType = [](MPI_Datatype& dt) {
        if (dt != MPI_DATATYPE_NULL) MPI_Type_free(&dt);
    };
    freeType(ddtype_sendto_x_right);  freeType(ddtype_recvfrom_x_left);
    freeType(ddtype_sendto_x_left);   freeType(ddtype_recvfrom_x_right);
    freeType(ddtype_sendto_y_right);  freeType(ddtype_recvfrom_y_left);
    freeType(ddtype_sendto_y_left);   freeType(ddtype_recvfrom_y_right);
    freeType(ddtype_sendto_z_right);  freeType(ddtype_recvfrom_z_left);
    freeType(ddtype_sendto_z_left);   freeType(ddtype_recvfrom_z_right);
}

void MPISubdomain::makeGhostcellDDType() {
    int sizes[3] = { nz_sub+1, ny_sub+1, nx_sub+1 };
    int subs[3], starts[3];

    auto createAndCommit = [&](MPI_Datatype& dtype) {
        MPI_Type_create_subarray(3, sizes, subs, starts, MPI_ORDER_C, MPI_DOUBLE, &dtype);
        MPI_Type_commit(&dtype);
    };

    // X direction
    subs[0] = nz_sub+1; subs[1] = ny_sub+1; subs[2] = 1;
    starts[0]=0; starts[1]=0; starts[2]=nx_sub-1; createAndCommit(ddtype_sendto_x_right);
    starts[2]=0;                                   createAndCommit(ddtype_recvfrom_x_left);
    starts[2]=1;                                   createAndCommit(ddtype_sendto_x_left);
    starts[2]=nx_sub;                              createAndCommit(ddtype_recvfrom_x_right);

    // Y direction
    subs[0] = nz_sub+1; subs[1] = 1; subs[2] = nx_sub+1;
    starts[0]=0; starts[2]=0;
    starts[1]=ny_sub-1; createAndCommit(ddtype_sendto_y_right);
    starts[1]=0;        createAndCommit(ddtype_recvfrom_y_left);
    starts[1]=1;        createAndCommit(ddtype_sendto_y_left);
    starts[1]=ny_sub;   createAndCommit(ddtype_recvfrom_y_right);

    // Z direction
    subs[0] = 1; subs[1] = ny_sub+1; subs[2] = nx_sub+1;
    starts[1]=0; starts[2]=0;
    starts[0]=nz_sub-1; createAndCommit(ddtype_sendto_z_right);
    starts[0]=0;        createAndCommit(ddtype_recvfrom_z_left);
    starts[0]=1;        createAndCommit(ddtype_sendto_z_left);
    starts[0]=nz_sub;   createAndCommit(ddtype_recvfrom_z_right);
}

void MPISubdomain::ghostcellUpdate(std::vector<double>& theta,
                                   const CartComm1D& cx, const CartComm1D& cy,
                                   const CartComm1D& cz, const GlobalParams&) {
    // Host-side path — uses derived subarray datatypes on a host buffer
    // (fast on host: MPI packs the strided face into a contiguous send buf
    //  via memcpy). Called once after initialization to fill ghosts of the
    //  initial theta on rank 0 before the H2D copy.
    double* h_theta = theta.data();
    MPI_Request reqs[12];
    int r = 0;
    MPI_Isend(h_theta, 1, ddtype_sendto_x_right,   cx.east_rank, 111, cx.comm, &reqs[r++]);
    MPI_Irecv(h_theta, 1, ddtype_recvfrom_x_left,  cx.west_rank, 111, cx.comm, &reqs[r++]);
    MPI_Isend(h_theta, 1, ddtype_sendto_x_left,    cx.west_rank, 222, cx.comm, &reqs[r++]);
    MPI_Irecv(h_theta, 1, ddtype_recvfrom_x_right, cx.east_rank, 222, cx.comm, &reqs[r++]);
    MPI_Isend(h_theta, 1, ddtype_sendto_y_right,   cy.east_rank, 333, cy.comm, &reqs[r++]);
    MPI_Irecv(h_theta, 1, ddtype_recvfrom_y_left,  cy.west_rank, 333, cy.comm, &reqs[r++]);
    MPI_Isend(h_theta, 1, ddtype_sendto_y_left,    cy.west_rank, 444, cy.comm, &reqs[r++]);
    MPI_Irecv(h_theta, 1, ddtype_recvfrom_y_right, cy.east_rank, 444, cy.comm, &reqs[r++]);
    MPI_Isend(h_theta, 1, ddtype_sendto_z_right,   cz.east_rank, 555, cz.comm, &reqs[r++]);
    MPI_Irecv(h_theta, 1, ddtype_recvfrom_z_left,  cz.west_rank, 555, cz.comm, &reqs[r++]);
    MPI_Isend(h_theta, 1, ddtype_sendto_z_left,    cz.west_rank, 666, cz.comm, &reqs[r++]);
    MPI_Irecv(h_theta, 1, ddtype_recvfrom_z_right, cz.east_rank, 666, cz.comm, &reqs[r++]);
    MPI_Waitall(r, reqs, MPI_STATUSES_IGNORE);
}

// `ghostcellUpdateDevice`, `allocGhostBufsDevice`, `freeGhostBufsDevice`
// are defined in `ghostcell_cuda.cu` to keep CUDA out of this translation unit.

void MPISubdomain::indices(const GlobalParams&,
                           int rankx, int npx, int ranky, int npy, int rankz, int npz) {
    std::fill(theta_x_left_index.begin(),  theta_x_left_index.end(),  0);
    std::fill(theta_x_right_index.begin(), theta_x_right_index.end(), 0);
    if (rankx == 0)     theta_x_left_index[1]        = 1;
    if (rankx == npx-1) theta_x_right_index[nx_sub-1] = 1;

    std::fill(theta_y_left_index.begin(),  theta_y_left_index.end(),  0);
    std::fill(theta_y_right_index.begin(), theta_y_right_index.end(), 0);
    if (ranky == 0)     theta_y_left_index[1]        = 1;
    if (ranky == npy-1) theta_y_right_index[ny_sub-1] = 1;

    std::fill(theta_z_left_index.begin(),  theta_z_left_index.end(),  0);
    std::fill(theta_z_right_index.begin(), theta_z_right_index.end(), 0);
    if (rankz == 0)     theta_z_left_index[1]        = 1;
    if (rankz == npz-1) theta_z_right_index[nz_sub-1] = 1;
}

void MPISubdomain::mesh(const GlobalParams& params,
                        int rankx, int ranky, int rankz,
                        int npx, int npy, int npz) {
    double ddx = params.lx / (params.nx - 1);
    for (int i = 0; i <= nx_sub; ++i) {
        if      (rankx == 0     && i == 0)      x_sub[i] = params.x0;
        else if (rankx == npx-1 && i == nx_sub) x_sub[i] = params.xN;
        else                                    x_sub[i] = params.x0 + ddx/2.0 + (ista - 2 + i) * ddx;
        dmx_sub[i] = ddx;
    }
    double ddy = params.ly / (params.ny - 1);
    for (int j = 0; j <= ny_sub; ++j) {
        if      (ranky == 0     && j == 0)      y_sub[j] = params.y0;
        else if (ranky == npy-1 && j == ny_sub) y_sub[j] = params.yN;
        else                                    y_sub[j] = params.y0 + ddy/2.0 + (jsta - 2 + j) * ddy;
        dmy_sub[j] = ddy;
    }
    double ddz = params.lz / (params.nz - 1);
    for (int k = 0; k <= nz_sub; ++k) {
        if      (rankz == 0     && k == 0)      z_sub[k] = params.z0;
        else if (rankz == npz-1 && k == nz_sub) z_sub[k] = params.zN;
        else                                    z_sub[k] = params.z0 + ddz/2.0 + (ksta - 2 + k) * ddz;
        dmz_sub[k] = ddz;
    }
}

void MPISubdomain::initialization(std::vector<double>& theta) {
    int nx1 = nx_sub + 1, ny1 = ny_sub + 1, nz1 = nz_sub + 1;
    for (int k = 0; k < nz1; ++k)
        for (int j = 0; j < ny1; ++j)
            for (int i = 0; i < nx1; ++i) {
                int idx = idx_ijk(i, j, k, nx1, ny1);
                theta[idx] = sin(Pi*x_sub[i]) * sin(Pi*y_sub[j]) * sin(Pi*z_sub[k])
                           + cos(Pi*x_sub[i]) * cos(Pi*y_sub[j]) * cos(Pi*z_sub[k]);
            }
}

void MPISubdomain::boundary(std::vector<double>& theta) {
    int nx1 = nx_sub + 1, ny1 = ny_sub + 1, nz1 = nz_sub + 1;

    for (int k = 0; k < nz1; ++k)
        for (int j = 0; j < ny1; ++j) {
            int jk = idx_jk(j, k, ny1);
            theta_x_left_sub[jk]  = theta[idx_ijk(0,     j, k, nx1, ny1)];
            theta_x_right_sub[jk] = theta[idx_ijk(nx1-1, j, k, nx1, ny1)];
        }
    for (int k = 0; k < nz1; ++k)
        for (int i = 0; i < nx1; ++i) {
            int ik = idx_ik(i, k, nx1);
            theta_y_left_sub[ik]  = theta[idx_ijk(i, 0,     k, nx1, ny1)];
            theta_y_right_sub[ik] = theta[idx_ijk(i, ny1-1, k, nx1, ny1)];
        }
    for (int j = 0; j < ny1; ++j)
        for (int i = 0; i < nx1; ++i) {
            int ij = idx_ij(i, j, nx1);
            theta_z_left_sub[ij]  = theta[idx_ijk(i, j, 0,     nx1, ny1)];
            theta_z_right_sub[ij] = theta[idx_ijk(i, j, nz1-1, nx1, ny1)];
        }
}
