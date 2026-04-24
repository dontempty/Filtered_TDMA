#include "MpiTopology.hpp"

#include <cstdio>
#include <stdexcept>

namespace channel {

MpiTopology::MpiTopology(MPI_Comm world,
                         int np1, int np2, int np3,
                         bool pbc1, bool pbc2, bool pbc3)
{
    int world_size = 0;
    MPI_Comm_size(world, &world_size);
    if (np1 * np2 * np3 != world_size) {
        if (int r; (MPI_Comm_rank(world, &r), r == 0)) {
            std::fprintf(stderr,
                "[MpiTopology] np1*np2*np3 (%d) != MPI world size (%d)\n",
                np1*np2*np3, world_size);
        }
        MPI_Abort(world, 1);
    }

    dims_[0] = np1; dims_[1] = np2; dims_[2] = np3;
    int periods[3] = { pbc1 ? 1 : 0, pbc2 ? 1 : 0, pbc3 ? 1 : 0 };
    periodic_[0] = pbc1; periodic_[1] = pbc2; periodic_[2] = pbc3;

    MPI_Cart_create(world, 3, dims_, periods, /*reorder=*/0, &cart_);
    MPI_Comm_rank(cart_, &rank_);
    MPI_Comm_size(cart_, &nprocs_);
    MPI_Cart_coords(cart_, rank_, 3, coords_);

    for (int a = 0; a < 3; ++a) {
        int remain[3] = {0, 0, 0};
        remain[a] = 1;
        MPI_Cart_sub(cart_, remain, &comm_axis_[a]);
        MPI_Comm_rank(comm_axis_[a], &sub_rank_[a]);
        MPI_Comm_size(comm_axis_[a], &sub_size_[a]);
        MPI_Cart_shift(cart_, a, 1, &left_[a], &right_[a]);
        MPI_Cart_shift(comm_axis_[a], 0, 1, &sub_left_[a], &sub_right_[a]);
    }

    // xy-plane sub-communicator (all ranks with the same iz coordinate)
    int remain_xy[3] = {1, 1, 0};
    MPI_Cart_sub(cart_, remain_xy, &comm_xy_);

    // xz-plane sub-communicator (all ranks with the same iy coordinate)
    // key = ix_coord + iz_coord*np1 ensures sequential z-block ordering for PaScaL_TDMA
    int color_xz = coords_[1];
    int key_xz   = coords_[0] + coords_[2] * dims_[0];
    MPI_Comm_split(cart_, color_xz, key_xz, &comm_xz_);
    MPI_Comm_rank(comm_xz_, &rank_xz_);
    MPI_Comm_size(comm_xz_, &size_xz_);
}

MpiTopology::~MpiTopology()
{
    for (int a = 0; a < 3; ++a)
        if (comm_axis_[a] != MPI_COMM_NULL) MPI_Comm_free(&comm_axis_[a]);
    if (comm_xy_ != MPI_COMM_NULL) MPI_Comm_free(&comm_xy_);
    if (comm_xz_ != MPI_COMM_NULL) MPI_Comm_free(&comm_xz_);
    if (cart_ != MPI_COMM_NULL) MPI_Comm_free(&cart_);
}

void MpiTopology::print() const
{
    if (rank_ != 0) return;
    std::printf("====== MpiTopology ======\n");
    std::printf("  dims=(%d,%d,%d)  total=%d\n",
                dims_[0], dims_[1], dims_[2], nprocs_);
    std::printf("  periodic=(%d,%d,%d)\n", periodic_[0], periodic_[1], periodic_[2]);
    std::printf("=========================\n");
}

} // namespace channel
