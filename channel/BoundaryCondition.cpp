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

    // Wall BC — MPM-STD-style "zero ghost" (UBCzbt = UBCzup = 0):
    //   * U, V (cell-centered in z): set ghost cells to 0 (NOT antisymmetric).
    //     Combined with the wall-flag drop in adi_sweep_z_ (Az=0 / Cz=0 with
    //     NO fold), this matches MPM-STD's `kum=kup=0` treatment exactly.
    //   * W (face-centered in z): wall faces (k=1, k=nz+1) = 0.
    //
    // Why not antisymmetric: antisymm gives a slightly stiffer wall-shear
    // discretization (extra +nu/dz²·U(1) in M·U^n at k=1) compared to
    // MPM-STD's flag-drop, which over-damps wall-layer fluctuations and
    // pushes the flow back to laminar Poiseuille at sub-critical Re.
    if (at_low_wall_) {
        for (int j = 0; j <= ny + 1; ++j)
            for (int i = 0; i <= nx + 1; ++i) {
                U(i, j, 0) = 0.0;      // zero ghost (MPM-STD convention)
                V(i, j, 0) = 0.0;
                W(i, j, 1) = 0.0;      // bottom wall face
            }
    }
    if (at_high_wall_) {
        for (int j = 0; j <= ny + 1; ++j)
            for (int i = 0; i <= nx + 1; ++i) {
                U(i, j, nz + 1) = 0.0;
                V(i, j, nz + 1) = 0.0;
                W(i, j, nz + 1) = 0.0; // top wall face
            }
    }
}

} // namespace channel
