// channel/Statistics.hpp
//
// Time-averaged z-profile statistics for channel flow.
//
// accumulate(): zero MPI — local Welford incremental mean update per z-slab.
//   Per sample: xy-plane mean of u, u², v, v², wc, wc², u·wc, p.
//
// write(): 8 MPI_Allreduce calls only (all ranks must call).
//   Computes:
//     u_tau  = sqrt(nu · |<U(z=0)>| / zc[0])
//     u_rms  = sqrt(<u²> - <u>²)
//     uw_stress = <u·wc> - <u><wc>
//   Outputs Tecplot ASCII: Z, Z+, U_mean, W_mean, u_rms, v_rms, w_rms,
//                           uw_stress, P_mean

#ifndef CHANNEL_STATISTICS_HPP
#define CHANNEL_STATISTICS_HPP

#include <string>
#include <vector>

#include "Field.hpp"

namespace channel {

class MpiTopology;
class Subdomain;
class Grid;

class Statistics {
public:
    Statistics(const MpiTopology& topo,
               const Subdomain& sub,
               const Grid& grid);

    // No MPI: pure local Welford update.
    void accumulate(const Field<double>& U,
                    const Field<double>& V,
                    const Field<double>& W,
                    const Field<double>& P);

    // All ranks must call (MPI_Allreduce inside). Rank 0 writes Tecplot file.
    // Re_b is used to compute nu = 1/Re_b for z+ and u_tau.
    void write(const std::string& path, int step, double Re_b,
               bool reset_after = false);

    void reset();
    long samples() const { return n_; }

private:
    const MpiTopology& topo_;
    const Subdomain&   sub_;
    const Grid&        grid_;

    int  nz_global_ = 0;
    int  nz_local_  = 0;
    int  kstart_    = 0;   // 0-based global index of this rank's first cell
    long n_         = 0;

    // Local Welford running means (size = nz_local_)
    std::vector<double> U_m_;    // <u>_xy
    std::vector<double> U2_m_;   // <u²>_xy
    std::vector<double> V_m_;    // <v>_xy
    std::vector<double> V2_m_;   // <v²>_xy
    std::vector<double> Wc_m_;   // <wc>_xy  (W face → cell-centre)
    std::vector<double> Wc2_m_;  // <wc²>_xy
    std::vector<double> UWc_m_;  // <u·wc>_xy
    std::vector<double> P_m_;    // <p>_xy

    // Global cell-centre z-coordinates (0-indexed, size = nz_global_)
    std::vector<double> zc_global_;

    void gather_to_global(const std::vector<double>& local,
                          std::vector<double>& global) const;
};

} // namespace channel

#endif // CHANNEL_STATISTICS_HPP
