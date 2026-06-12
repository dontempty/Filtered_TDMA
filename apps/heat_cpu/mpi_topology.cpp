#include "mpi_topology.hpp"

void MPITopology::make() {
    int per[3] = { int(periods_[0]), int(periods_[1]), int(periods_[2]) };
    if (MPI_Cart_create(MPI_COMM_WORLD, 3, dims_.data(), per, 0, &world_cart_) != MPI_SUCCESS)
        throw std::runtime_error("MPI_Cart_create failed");
    defineSubcomm(0, comm_x_);
    defineSubcomm(1, comm_y_);
    defineSubcomm(2, comm_z_);
}

void MPITopology::clean() {
    if (world_cart_ != MPI_COMM_NULL) MPI_Comm_free(&world_cart_);
}

void MPITopology::defineSubcomm(int dim, CartComm1D& sub) {
    int remain[3] = {0, 0, 0};
    remain[dim] = 1;
    MPI_Cart_sub(world_cart_, remain, &sub.comm);
    MPI_Comm_rank(sub.comm, &sub.myrank);
    MPI_Comm_size(sub.comm, &sub.nprocs);
    MPI_Cart_shift(sub.comm, 0, 1, &sub.west_rank, &sub.east_rank);
}
