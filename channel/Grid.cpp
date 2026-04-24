#include "Grid.hpp"

#include "Config.hpp"
#include "MpiTopology.hpp"
#include "Subdomain.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace channel {

namespace {

// Face position xf[i] for i = 0..n_global (inclusive), normalized to [0,L].
std::vector<double> face_positions(int n_global, int uniform, double gamma, double L)
{
    std::vector<double> xf(static_cast<std::size_t>(n_global) + 1);
    if (uniform == 1 || std::fabs(gamma) < 1.0e-14) {
        for (int i = 0; i <= n_global; ++i)
            xf[i] = L * static_cast<double>(i) / static_cast<double>(n_global);
    } else {
        const double th = std::tanh(gamma);
        for (int i = 0; i <= n_global; ++i) {
            double zeta = static_cast<double>(i) / static_cast<double>(n_global);
            xf[i] = 0.5 * L * (1.0 - std::tanh(gamma * (1.0 - 2.0 * zeta)) / th);
        }
    }
    return xf;
}

} // namespace

Grid::Grid(const Config& cfg, const MpiTopology& topo, const Subdomain& sub)
{
    L_[0] = cfg.Lx; L_[1] = cfg.Ly; L_[2] = cfg.Lz;

    int    uni[3]   = { cfg.uniform1, cfg.uniform2, cfg.uniform3 };
    double gam[3]   = { cfg.gamma1,   cfg.gamma2,   cfg.gamma3   };
    bool   per[3]   = { cfg.pbc1,     cfg.pbc2,     cfg.pbc3     };

    for (int a = 0; a < 3; ++a) {
        build_axis(a, sub.global_n(a), sub.ista(a), sub.iend(a),
                   uni[a], gam[a], L_[a], per[a]);
    }
    (void)topo;
}

void Grid::build_axis(int axis,
                      int n_global, int ista, int iend,
                      int uniform, double gamma, double L_axis,
                      bool periodic)
{
    // Build full global face positions, derive cell-centers, then slice locally.
    auto xf = face_positions(n_global, uniform, gamma, L_axis);

    std::vector<double> xc_g(static_cast<std::size_t>(n_global) + 2);
    for (int i = 1; i <= n_global; ++i)
        xc_g[i] = 0.5 * (xf[i - 1] + xf[i]);
    // Halo cell centers: mirror or extrapolate
    if (periodic) {
        xc_g[0]            = xc_g[n_global] - L_axis;
        xc_g[n_global + 1] = xc_g[1]        + L_axis;
    } else {
        xc_g[0]            = xf[0]        - (xc_g[1]        - xf[0]);
        xc_g[n_global + 1] = xf[n_global] + (xf[n_global]   - xc_g[n_global]);
    }

    // dx_g[i] = xf[i] - xf[i-1]   for i=1..n_global; ghosts = neighbor copy
    std::vector<double> dx_g(static_cast<std::size_t>(n_global) + 2);
    for (int i = 1; i <= n_global; ++i) dx_g[i] = xf[i] - xf[i - 1];
    dx_g[0]            = dx_g[periodic ? n_global : 1];
    dx_g[n_global + 1] = dx_g[periodic ? 1        : n_global];

    // dmx_g[i] = xc[i] - xc[i-1]  for i=1..n_global+1
    std::vector<double> dmx_g(static_cast<std::size_t>(n_global) + 2);
    for (int i = 1; i <= n_global + 1; ++i) dmx_g[i] = xc_g[i] - xc_g[i - 1];
    dmx_g[0] = dmx_g[periodic ? n_global : 1];

    // Slice local: indices ista..iend become 1..n; ghost from neighbor cells
    int n_local = iend - ista + 1;
    auto& xL   = x_[axis];   xL.resize(static_cast<std::size_t>(n_local) + 2);
    auto& dxL  = dx_[axis];  dxL.resize(static_cast<std::size_t>(n_local) + 2);
    auto& dmxL = dmx_[axis]; dmxL.resize(static_cast<std::size_t>(n_local) + 2);

    for (int i = 0; i <= n_local + 1; ++i) {
        int g = ista - 1 + i;          // global index
        if (g < 0) g = periodic ? g + n_global : 0;
        if (g > n_global + 1) g = periodic ? g - n_global : n_global + 1;
        xL[i]   = xc_g[g];
        dxL[i]  = dx_g[g];
        dmxL[i] = dmx_g[g];
    }
}

void Grid::print() const
{
    auto print_ax = [](const char* name, const std::vector<double>& a) {
        std::printf("  %s[0..%zu]: ", name, a.size() - 1);
        std::size_t n = a.size();
        std::size_t lim = std::min<std::size_t>(n, 5);
        for (std::size_t i = 0; i < lim; ++i) std::printf("%g ", a[i]);
        if (n > 6) std::printf("... ");
        if (n > 5) std::printf("%g", a.back());
        std::printf("\n");
    };
    std::printf("====== Grid (rank-local) ======\n");
    const char* axes[3] = { "x", "y", "z" };
    for (int a = 0; a < 3; ++a) {
        std::printf(" axis %s (L=%g):\n", axes[a], L_[a]);
        print_ax("x  ", x_[a]);
        print_ax("dx ", dx_[a]);
        print_ax("dmx", dmx_[a]);
    }
    std::printf("===============================\n");
}

} // namespace channel
