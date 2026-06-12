// channel/ChannelForcing.hpp
//
// Bulk-flow forcing for channel: maintains either a constant mean
// pressure gradient (mode = PRESSURE_GRADIENT) or a constant bulk
// velocity (mode = MASS_FLOW). The latter follows MPM-STD's
// `cuda_momentum_masscorrection` pattern: at each time step the
// pressure gradient is updated as
//
//     dPdx_new = dPdx_old + (Q_target - Q_pseudo) / dt
//
// where Q_pseudo = ⟨U⟩_volume of the pseudo-velocity. Then U is shifted
// by -dt · (dPdx_new - dPdx_old) so the bulk velocity hits the target.

#ifndef CHANNEL_FORCING_HPP
#define CHANNEL_FORCING_HPP

#include "Config.hpp"
#include "Field.hpp"

namespace channel {

class MpiTopology;
class Subdomain;
class Grid;

class ChannelForcing {
public:
    ChannelForcing(const Config& cfg,
                   const MpiTopology& topo,
                   const Subdomain& sub,
                   const Grid& grid);

    /// Update internal dPdx and shift U so that ⟨U⟩ matches the target.
    /// Returns the (signed) mean pressure gradient currently applied.
    /// In PRESSURE_GRADIENT mode this is just the configured constant.
    double correct(Field<double>& U, double dt);

    /// Read-only accessor for the running dPdx (e.g. for I/O & restart).
    double mean_dPdx() const { return dPdx_; }
    void   set_mean_dPdx(double v) { dPdx_ = v; }

private:
    double bulk_velocity_(const Field<double>& U) const;

    const Config&      cfg_;
    const MpiTopology& topo_;
    const Subdomain&   sub_;
    const Grid&        grid_;

    double dPdx_         = 0.0;
    double total_volume_ = 0.0;   // global ∫ dV
};

} // namespace channel

#endif // CHANNEL_FORCING_HPP
