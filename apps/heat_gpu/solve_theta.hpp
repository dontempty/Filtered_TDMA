#ifndef SOLVE_THETA_HPP
#define SOLVE_THETA_HPP

#include <vector>
#include "global.hpp"
#include "mpi_topology.hpp"
#include "mpi_subdomain.hpp"

class SolveTheta {
public:
    SolveTheta(const GlobalParams& params,
               const MPITopology& topo,
               MPISubdomain& sub);

    /// GPU-native time-step loop for the 3D heat ADI solver.
    /// `theta` is host-side (read on entry, written on exit); all per-step
    /// work runs on device with persistent buffers — only one H2D at the
    /// start and one D2H at the end.
    void run(std::vector<double>& theta);
    void profile(std::vector<double>& theta);  // alias of run() for now

private:
    const GlobalParams& params_;
    const MPITopology&  topo_;
    MPISubdomain&       sub_;
};

#endif // SOLVE_THETA_HPP
