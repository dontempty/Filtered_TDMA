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

    // Wall BC — v2 style:
    //   * U, V (cell-centered in z): antisymmetric ghost (ghost = -interior).
    //     Pairs with mirror grid (dmz[1] = dz[1]) and a diagonal fold in the
    //     z ADI to give the correct 3·ν/dz² wall-row coefficient.
    //   * W (face-centered in z): wall faces (k=1, k=nz+1) = 0.
    if (at_low_wall_) {
        for (int j = 0; j <= ny + 1; ++j)
            for (int i = 0; i <= nx + 1; ++i) {
                U(i, j, 0) = -U(i, j, 1);   // antisymmetric ghost
                V(i, j, 0) = -V(i, j, 1);
                W(i, j, 1) =  0.0;          // bottom wall face
            }
    }
    if (at_high_wall_) {
        for (int j = 0; j <= ny + 1; ++j)
            for (int i = 0; i <= nx + 1; ++i) {
                U(i, j, nz + 1) = -U(i, j, nz);
                V(i, j, nz + 1) = -V(i, j, nz);
                W(i, j, nz + 1) =  0.0;     // top wall face
            }
    }
}

} // namespace channel
