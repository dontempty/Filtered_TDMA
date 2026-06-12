// channel/MpiTopology.hpp
//
// 3D Cartesian MPI topology + 1D direction sub-communicators.
//
// PaScaL_TCS analogue: module_mpi_topology.f90
//   comm_world          ↔ cart_comm
//   comm_1d_x1, x2, x3  ↔ comm_x, comm_y, comm_z
//   nrank_x1, nx1prev, nx1next  ↔  rank_x, left_x, right_x  (etc.)
//
// Composite communicators for pressure Poisson data transposes
// (analogue of comm_1d_x1n2) are deferred to PressureSolver (P3).

#ifndef CHANNEL_MPI_TOPOLOGY_HPP
#define CHANNEL_MPI_TOPOLOGY_HPP

#include <mpi.h>

namespace channel {

class MpiTopology {
public:
    /// Build a (np1 × np2 × np3) Cartesian topology over `world`.
    /// Periodic flags map to (pbc1, pbc2, pbc3).
    MpiTopology(MPI_Comm world,
                int np1, int np2, int np3,
                bool pbc1, bool pbc2, bool pbc3);

    ~MpiTopology();

    MpiTopology(const MpiTopology&)            = delete;
    MpiTopology& operator=(const MpiTopology&) = delete;

    // ---- accessors ------------------------------------------------------
    MPI_Comm cart() const { return cart_; }

    MPI_Comm comm(int axis) const { return comm_axis_[axis]; }
    MPI_Comm comm_x()  const { return comm_axis_[0]; }
    MPI_Comm comm_y()  const { return comm_axis_[1]; }
    MPI_Comm comm_z()  const { return comm_axis_[2]; }
    MPI_Comm comm_xy() const { return comm_xy_; }   ///< np1*np2 ranks, same iz
    MPI_Comm comm_xz() const { return comm_xz_; }   ///< np1*np3 ranks, same iy
    int rank_xz()      const { return rank_xz_; }
    int size_xz()      const { return size_xz_; }

    int rank()         const { return rank_;   }
    int nprocs()       const { return nprocs_; }
    int dim(int a)     const { return dims_[a];   }
    int coord(int a)   const { return coords_[a]; }

    int rank_in(int axis)        const { return sub_rank_[axis];      }
    int size_in(int axis)        const { return sub_size_[axis];      }
    int left_in(int axis)        const { return left_[axis];           }
    int right_in(int axis)       const { return right_[axis];          }

    /// Neighbour ranks within the 1-D sub-communicator (for FilteredTdmaSolver).
    int left_in_sub(int axis)    const { return sub_left_[axis];       }
    int right_in_sub(int axis)   const { return sub_right_[axis];      }

    bool periodic(int axis) const { return periodic_[axis]; }

    /// pretty-print on rank 0
    void print() const;

private:
    MPI_Comm cart_                = MPI_COMM_NULL;
    MPI_Comm comm_axis_[3]        = {MPI_COMM_NULL, MPI_COMM_NULL, MPI_COMM_NULL};
    MPI_Comm comm_xy_             = MPI_COMM_NULL;
    MPI_Comm comm_xz_             = MPI_COMM_NULL;
    int      rank_xz_             = 0;
    int      size_xz_             = 1;
    int      dims_[3]             = {1, 1, 1};
    int      coords_[3]           = {0, 0, 0};
    int      sub_rank_[3]         = {0, 0, 0};
    int      sub_size_[3]         = {1, 1, 1};
    int      left_[3]             = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
    int      right_[3]            = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
    int      sub_left_[3]         = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
    int      sub_right_[3]        = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
    bool     periodic_[3]         = {true, true, false};
    int      rank_                = 0;
    int      nprocs_              = 1;
};

} // namespace channel

#endif // CHANNEL_MPI_TOPOLOGY_HPP
