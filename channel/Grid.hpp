// channel/Grid.hpp
//
// Per-rank 1D coordinate arrays for the staggered grid.
//
// For each axis we keep three arrays (size = local_n + 2, halo included):
//   x[i]   : cell-center coordinate
//   dx[i]  : cell width   (face-to-face)
//   dmx[i] : center-to-center distance
//
// For wall-normal (z), if cfg.uniform3 != 1 a hyperbolic-tangent stretch
// concentrates cells near both walls (PaScaL_TCS-style):
//   xf(zeta) = 0.5*Lz * (1 - tanh(gamma*(1-2*zeta))/tanh(gamma))
//
// PaScaL_TCS analogue: module_mpi_subdomain.f90:160-295

#ifndef CHANNEL_GRID_HPP
#define CHANNEL_GRID_HPP

#include <vector>

namespace channel {

class MpiTopology;
class Subdomain;
struct Config;

class Grid {
public:
    Grid(const Config& cfg, const MpiTopology& topo, const Subdomain& sub);

    // Local 1D arrays per axis. Size = sub.n(axis) + 2 (halos at 0 and n+1).
    const std::vector<double>& x  (int axis) const { return x_[axis];   }
    const std::vector<double>& dx (int axis) const { return dx_[axis];  }
    const std::vector<double>& dmx(int axis) const { return dmx_[axis]; }

    double L(int axis) const { return L_[axis]; }

    void print() const;

private:
    void build_axis(int axis,
                    int n_global, int ista, int iend,
                    int uniform, double gamma, double L_axis,
                    bool periodic);

    std::vector<double> x_[3];
    std::vector<double> dx_[3];
    std::vector<double> dmx_[3];
    double L_[3] = {0.0, 0.0, 0.0};
};

} // namespace channel

#endif // CHANNEL_GRID_HPP
