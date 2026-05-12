#ifndef MPI_TOPOLOGY_HPP
#define MPI_TOPOLOGY_HPP

#include <mpi.h>
#include <array>
#include <stdexcept>

struct CartComm1D {
    int      myrank, nprocs, west_rank, east_rank;
    MPI_Comm comm;
};

class MPITopology {
public:
    MPITopology() = default;

    void init(const std::array<int, 3>& dims,
              const std::array<bool, 3>& periods) {
        dims_    = dims;
        periods_ = periods;
        world_cart_ = MPI_COMM_NULL;
    }

    void make();
    void clean();

    MPI_Comm            worldCart() const { return world_cart_; }
    const CartComm1D&   commX()    const { return comm_x_; }
    const CartComm1D&   commY()    const { return comm_y_; }
    const CartComm1D&   commZ()    const { return comm_z_; }

private:
    void defineSubcomm(int dim, CartComm1D& sub);
    std::array<int, 3>   dims_;
    std::array<bool, 3>  periods_;
    MPI_Comm             world_cart_ = MPI_COMM_NULL;
    CartComm1D           comm_x_, comm_y_, comm_z_;
};

#endif // MPI_TOPOLOGY_HPP
