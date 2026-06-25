#ifndef GLOBAL_PARAMS_HPP
#define GLOBAL_PARAMS_HPP

#include <array>
#include <string>

class GlobalParams {
public:
    GlobalParams() = default;
    void load(const std::string& filename);

    // Physical / numerical parameters
    double Tmax, dt;
    double rho, eps_constant;
    int Nt;
    int Nt_warmup;

    int nx, ny, nz;
    int nxm, nym, nzm;

    double lx, ly, lz;
    double x0, xN, y0, yN, z0, zN;
    double dx, dy, dz;

    // MPI process decomposition
    std::array<int, 3> np_dim;

    // Run mode: "order" or "strong"
    std::string option;

    // TDMA backend: "filtered" (default) or "pascal"
    std::string tdma_backend = "filtered";

    // Per-direction boundary: false = Dirichlet wall, true = periodic.
    // Set via input keys `periodic` (all three) or `periodic_x/y/z`.
    std::array<bool, 3> periodic = {false, false, false};
};

#endif // GLOBAL_PARAMS_HPP
