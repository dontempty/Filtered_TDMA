#include "HaloExchanger.hpp"

#include "MpiTopology.hpp"
#include "Subdomain.hpp"

#include <array>

namespace channel {

namespace {

// Build a subarray DDT for a single face plane.
// `sizes`  = full storage extent {nxt, nyt, nzt}, ordered (axis0, axis1, axis2)
// `subs`   = subarray extent for the face slab
// `starts` = starting indices within the full array
// Stored in C order (last index fastest in MPI_ORDER_C); but the field
// is x-fastest, so we declare the array to MPI in Fortran order to match.
MPI_Datatype make_face_type(const std::array<int,3>& sizes,
                            const std::array<int,3>& subs,
                            const std::array<int,3>& starts)
{
    MPI_Datatype t;
    MPI_Type_create_subarray(3,
                             sizes.data(),
                             subs.data(),
                             starts.data(),
                             MPI_ORDER_FORTRAN,
                             MPI_DOUBLE,
                             &t);
    MPI_Type_commit(&t);
    return t;
}

} // namespace

HaloExchanger::HaloExchanger(const MpiTopology& topo, const Subdomain& sub)
    : topo_(topo)
{
    nx_ = sub.nx();
    ny_ = sub.ny();
    nz_ = sub.nz();
    for (int a = 0; a < 3; ++a) build_subarray_(a, nx_, ny_, nz_);
}

HaloExchanger::~HaloExchanger()
{
    for (int a = 0; a < 3; ++a) {
        if (send_lo_[a] != MPI_DATATYPE_NULL) MPI_Type_free(&send_lo_[a]);
        if (send_hi_[a] != MPI_DATATYPE_NULL) MPI_Type_free(&send_hi_[a]);
        if (recv_lo_[a] != MPI_DATATYPE_NULL) MPI_Type_free(&recv_lo_[a]);
        if (recv_hi_[a] != MPI_DATATYPE_NULL) MPI_Type_free(&recv_hi_[a]);
    }
}

void HaloExchanger::build_subarray_(int axis, int nx, int ny, int nz)
{
    const std::array<int,3> sizes = { nx + 2, ny + 2, nz + 2 };
    std::array<int,3> subs   = { nx,     ny,     nz     };
    subs[axis] = 1;

    std::array<int,3> start_send_lo = { 1, 1, 1 };
    std::array<int,3> start_send_hi = { 1, 1, 1 };
    std::array<int,3> start_recv_lo = { 1, 1, 1 };
    std::array<int,3> start_recv_hi = { 1, 1, 1 };

    int n_axis = (axis == 0 ? nx : (axis == 1 ? ny : nz));
    start_send_lo[axis] = 1;
    start_send_hi[axis] = n_axis;
    start_recv_lo[axis] = 0;
    start_recv_hi[axis] = n_axis + 1;

    send_lo_[axis] = make_face_type(sizes, subs, start_send_lo);
    send_hi_[axis] = make_face_type(sizes, subs, start_send_hi);
    recv_lo_[axis] = make_face_type(sizes, subs, start_recv_lo);
    recv_hi_[axis] = make_face_type(sizes, subs, start_recv_hi);
}

void HaloExchanger::exchange_axis(Field<double>& f, int axis) const
{
    const int left  = topo_.left_in(axis);
    const int right = topo_.right_in(axis);
    const MPI_Comm comm = topo_.cart();

    constexpr int TAG_LO = 100;   // sent to left, received from right
    constexpr int TAG_HI = 200;   // sent to right, received from left

    MPI_Request req[4];
    int nreq = 0;

    // recv from right into hi-ghost
    MPI_Irecv(f.data(), 1, recv_hi_[axis], right, TAG_LO, comm, &req[nreq++]);
    // recv from left  into lo-ghost
    MPI_Irecv(f.data(), 1, recv_lo_[axis], left,  TAG_HI, comm, &req[nreq++]);
    // send lo-interior to left
    MPI_Isend(f.data(), 1, send_lo_[axis], left,  TAG_LO, comm, &req[nreq++]);
    // send hi-interior to right
    MPI_Isend(f.data(), 1, send_hi_[axis], right, TAG_HI, comm, &req[nreq++]);

    MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
}

void HaloExchanger::exchange(Field<double>& f) const
{
    exchange_axis(f, 0);
    exchange_axis(f, 1);
    exchange_axis(f, 2);
}

} // namespace channel
