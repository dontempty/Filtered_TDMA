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

    void run(std::vector<double>& theta);
    void profile(std::vector<double>& theta);

private:
    const GlobalParams& params_;
    const MPITopology&  topo_;
    MPISubdomain&       sub_;
};

#endif // SOLVE_THETA_HPP
