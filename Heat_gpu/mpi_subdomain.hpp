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

    /// Host-side ghost-cell exchange (used only during initialization).
    /// Operates on host `theta` via the derived-datatype subarray pattern.
    void ghostcellUpdate(std::vector<double>& theta,
                         const CartComm1D& cx, const CartComm1D& cy,
                         const CartComm1D& cz, const GlobalParams& params);

    /// Device-side ghost-cell exchange (called every timestep).
    /// Uses **pack → contiguous-buffer MPI → unpack** kernels so MPI sees
    /// only contiguous device pointers (CUDA-aware MPI fast path); matches
    /// the PaScaL_TDMA_F `ghostcell_update_cuda` pattern.
    /// Call `allocGhostBufsDevice` once after `make()` before invoking this.
    void ghostcellUpdateDevice(double* d_theta,
                               const CartComm1D& cx, const CartComm1D& cy,
                               const CartComm1D& cz);

    /// Allocate contiguous device send/recv buffers for the 6 faces of theta.
    /// Idempotent — safe to call more than once. Implementation in a `.cu`.
    void allocGhostBufsDevice();
    /// Free the device buffers; called from `clean()`.
    void freeGhostBufsDevice();

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

    // --- Device contiguous ghost-cell buffers (one pair per axis-direction) ---
    // Layout: each face packed as row-major. Size of each:
    //   x faces: (ny_sub+1) * (nz_sub+1)
    //   y faces: (nx_sub+1) * (nz_sub+1)
    //   z faces: (nx_sub+1) * (ny_sub+1)
    double* d_sbuf_x0 = nullptr;  double* d_sbuf_x1 = nullptr;
    double* d_rbuf_x0 = nullptr;  double* d_rbuf_x1 = nullptr;
    double* d_sbuf_y0 = nullptr;  double* d_sbuf_y1 = nullptr;
    double* d_rbuf_y0 = nullptr;  double* d_rbuf_y1 = nullptr;
    double* d_sbuf_z0 = nullptr;  double* d_sbuf_z1 = nullptr;
    double* d_rbuf_z0 = nullptr;  double* d_rbuf_z1 = nullptr;

private:
    MPI_Datatype ddtype_sendto_x_right,  ddtype_recvfrom_x_left;
    MPI_Datatype ddtype_sendto_x_left,   ddtype_recvfrom_x_right;
    MPI_Datatype ddtype_sendto_y_right,  ddtype_recvfrom_y_left;
    MPI_Datatype ddtype_sendto_y_left,   ddtype_recvfrom_y_right;
    MPI_Datatype ddtype_sendto_z_right,  ddtype_recvfrom_z_left;
    MPI_Datatype ddtype_sendto_z_left,   ddtype_recvfrom_z_right;
};

#endif // MPI_SUBDOMAIN_HPP
