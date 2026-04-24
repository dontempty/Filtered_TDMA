#include "BoundaryCondition.hpp"

#include "MpiTopology.hpp"
#include "Subdomain.hpp"

namespace channel {

BoundaryCondition::BoundaryCondition(const MpiTopology& topo, const Subdomain& sub)
    : topo_(topo), sub_(sub)
{
    const int Z = 2;
    if (!topo_.periodic(Z)) {
        at_low_wall_  = (topo_.rank_in(Z) == 0);
        at_high_wall_ = (topo_.rank_in(Z) == topo_.size_in(Z) - 1);
    }
}

void BoundaryCondition::apply(Field<double>& U,
                              Field<double>& V,
                              Field<double>& W) const
{
    const int nx = sub_.nx();
    const int ny = sub_.ny();
    const int nz = sub_.nz();

    // Lower wall  (k=0 is ghost; k=1 is the bottom wall W-face).
    // U, V: antisymmetric ghost to enforce zero at the wall without
    //        modifying interior cell values.
    // W:    bottom wall face (k=1) must be identically 0; no meaningful
    //        ghost is needed below it (W(0) is irrelevant for the stencil
    //        because W(1)=0 itself cancels the neighbor term).
    if (at_low_wall_) {
        for (int j = 0; j <= ny + 1; ++j)
            for (int i = 0; i <= nx + 1; ++i) {
                U(i, j, 0) = -U(i, j, 1);   // antisymmetric ghost
                V(i, j, 0) = -V(i, j, 1);   // antisymmetric ghost
                W(i, j, 1) = 0.0;            // bottom wall face = 0
            }
    }

    // Upper wall  (k=nz+1 is the top wall W-face AND the ghost slot for U/V).
    // W faces layout (1-indexed, size nz+2 including ghosts):
    //   k=1          : bottom wall face  (= 0, set above)
    //   k=2..nz      : interior faces    (must NOT be zeroed here)
    //   k=nz+1       : top wall face     (= 0)
    // U, V layout: cell centres k=1..nz; ghost at k=nz+1.
    if (at_high_wall_) {
        for (int j = 0; j <= ny + 1; ++j)
            for (int i = 0; i <= nx + 1; ++i) {
                U(i, j, nz + 1) = -U(i, j, nz);   // antisymmetric ghost
                V(i, j, nz + 1) = -V(i, j, nz);   // antisymmetric ghost
                W(i, j, nz + 1) = 0.0;             // top wall face = 0
                // W(i,j,nz) is an interior face — must NOT be zeroed
            }
    }
}

} // namespace channel
