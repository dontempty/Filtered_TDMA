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

    // ---- Cell-center stats (clean: no z-halo dependence) ----
    // Used for u_rms, v_rms, w_rms, mean U/V/Wc, P_mean.
    std::vector<double> U_m_;    // <u>_xy at zc[k]
    std::vector<double> U2_m_;   // <u²>_xy at zc[k]
    std::vector<double> V_m_;    // <v>_xy at zc[k]
    std::vector<double> V2_m_;   // <v²>_xy at zc[k]
    std::vector<double> Wc_m_;   // <wc>_xy at zc[k] (cell-center W from face interp)
    std::vector<double> Wc2_m_;  // <wc²>_xy at zc[k]
    std::vector<double> P_m_;    // <p>_xy at zc[k]

    // ---- Corner stats (clean from PaScaL rank-boundary bias) ----
    // Used for uw cross stat only. Stored at z-face zf[k].
    // Output uw at zc[k] = interpolation of corner uw at zf[k] and zf[k+1].
    std::vector<double> Ug_m_;   // <Ug>_xy at zf[k]
    std::vector<double> Wg_m_;   // <Wg>_xy at zf[k]
    std::vector<double> UWg_m_;  // <Ug·Wg>_xy at zf[k]

    // Global cell-centre z-coordinates (0-indexed, size = nz_global_) — kept for reference
    std::vector<double> zc_global_;

    // Global z-FACE positions (lower face of each global cell), size = nz_global_
    // Stats are output at these positions (MPM-STD corner-stat convention).
    std::vector<double> z_face_global_;

    void gather_to_global(const std::vector<double>& local,
                          std::vector<double>& global) const;
};

} // namespace channel

#endif // CHANNEL_STATISTICS_HPP
