#ifndef CHANNEL_TDMA_SOLVER_GPU_HPP
#define CHANNEL_TDMA_SOLVER_GPU_HPP

#include <memory>
#include <mpi.h>
#include <string>

#include "filtered_tdma_cuda.hpp"
#include "pascal_tdma_many_cuda.hpp"

namespace channel {

class MpiTopology;

class TdmaSolverGPU {
public:
    enum class Backend { FILTERED, PASCAL };

    static Backend parse_backend(const std::string& s);

    TdmaSolverGPU(const MpiTopology& topo, int axis,
                  int n_sys, int n_row, bool periodic,
                  Backend backend, double eps_constant = 1.0e-12);

    void solve(double* d_A, double* d_B, double* d_C, double* d_D);
    void solve_cycl(double* d_A, double* d_B, double* d_C, double* d_D);
    void set_rho_device(const double* d_A, const double* d_B, const double* d_C);
    void set_eps_constant(double eps_constant);

    double last_comm_ms() const;
    double last_gpu_ms() const;

private:
    Backend backend_;
    int n_sys_ = 0;
    int n_row_ = 0;
    bool periodic_ = false;
    std::unique_ptr<FilteredTDMACUDA> filt_;
    std::unique_ptr<PaScaLTDMAManyCUDA> pasc_;
};

} // namespace channel

#endif // CHANNEL_TDMA_SOLVER_GPU_HPP
