// channel/BoundaryCondition.hpp
//
// Apply velocity Dirichlet conditions at the wall-normal (z) walls.
// Periodic conditions in x and y are realized by HaloExchanger and so
// require no action here.
//
// On a staggered grid:
//   U lives at x-face centers   → wall (z) ghost holds an image cell
//   V lives at y-face centers   → wall (z) ghost holds an image cell
//   W lives at z-face centers   → wall is at k=1 (lower) and k=n3 (upper);
//                                  W at the wall is set to 0 directly
//                                  (Dirichlet on the velocity dof).
//
// PaScaL_TCS analogue: subroutines mpi_momentum_blockLdU / V / W and
// the BC handling baked into the diffusion stencils.

#ifndef CHANNEL_BOUNDARY_CONDITION_HPP
#define CHANNEL_BOUNDARY_CONDITION_HPP

#include "Field.hpp"

namespace channel {

class MpiTopology;
class Subdomain;

class BoundaryCondition {
public:
    BoundaryCondition(const MpiTopology& topo, const Subdomain& sub);

    /// Apply no-slip wall BC at the wall-normal (z) walls.
    /// `U`, `V`: image-cell convention   ghost = -interior
    /// `W`     : direct Dirichlet at k=1 (lower wall) and k=nz (upper wall)
    void apply(Field<double>& U, Field<double>& V, Field<double>& W) const;

private:
    const MpiTopology& topo_;
    const Subdomain&   sub_;
    bool at_low_wall_  = false;   // this rank touches z=0
    bool at_high_wall_ = false;   // this rank touches z=Lz
};

} // namespace channel

#endif // CHANNEL_BOUNDARY_CONDITION_HPP
