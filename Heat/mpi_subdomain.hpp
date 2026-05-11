#ifndef MPI_SUBDOMAIN_HPP
#define MPI_SUBDOMAIN_HPP

#include <mpi.h>
#include <vector>
#include "global.hpp"
#include "mpi_topology.hpp"

extern const double Pi;

class MPISubdomain {
public:
    MPISubdomain();

    void make(const GlobalParams& params,
              int npx, int rankx, int npy, int ranky, int npz, int rankz);
    void clean();

    void makeGhostcellDDType();
    void ghostcellUpdate(std::vector<double>& theta,
                         const CartComm1D& cx, const CartComm1D& cy,
                         const CartComm1D& cz, const GlobalParams& params);

    void mesh(const GlobalParams& params,
              int rankx, int ranky, int rankz,
              int npx, int npy, int npz);
    void indices(const GlobalParams& params,
                 int rankx, int npx, int ranky, int npy, int rankz, int npz);
    void initialization(std::vector<double>& theta);
    void boundary(std::vector<double>& theta);

    // Subdomain dimensions
    int nx_sub, ny_sub, nz_sub;
    int ista, iend, jsta, jend, ksta, kend;

    // Coordinates and grid spacings
    std::vector<double> x_sub, y_sub, z_sub;
    std::vector<double> dmx_sub, dmy_sub, dmz_sub;

    // Boundary value buffers
    std::vector<double> theta_x_left_sub, theta_x_right_sub;
    std::vector<double> theta_y_left_sub, theta_y_right_sub;
    std::vector<double> theta_z_left_sub, theta_z_right_sub;

    // Boundary indicator arrays (1 at physical boundary, 0 elsewhere)
    std::vector<int> theta_x_left_index, theta_x_right_index;
    std::vector<int> theta_y_left_index, theta_y_right_index;
    std::vector<int> theta_z_left_index, theta_z_right_index;

private:
    MPI_Datatype ddtype_sendto_x_right,  ddtype_recvfrom_x_left;
    MPI_Datatype ddtype_sendto_x_left,   ddtype_recvfrom_x_right;
    MPI_Datatype ddtype_sendto_y_right,  ddtype_recvfrom_y_left;
    MPI_Datatype ddtype_sendto_y_left,   ddtype_recvfrom_y_right;
    MPI_Datatype ddtype_sendto_z_right,  ddtype_recvfrom_z_left;
    MPI_Datatype ddtype_sendto_z_left,   ddtype_recvfrom_z_right;
};

#endif // MPI_SUBDOMAIN_HPP
