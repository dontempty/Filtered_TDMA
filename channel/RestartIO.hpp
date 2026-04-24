// channel/RestartIO.hpp
//
// Binary restart files for channel flow, MPM-STD-compatible direct-access
// layout (one file per scalar field; rank 0 gathers the global array via
// MPI_Gatherv and writes contiguously).
//
// Files (under cfg.dir_cont_*):
//   cont_time.bin   : [time, dt, step (as double), dPdx]
//   cont_U.bin, cont_V.bin, cont_W.bin, cont_P.bin
//                   : Nx_g * Ny_g * Nz_g doubles, x-fastest

#ifndef CHANNEL_RESTART_IO_HPP
#define CHANNEL_RESTART_IO_HPP

#include <string>

#include "Field.hpp"

namespace channel {

class MpiTopology;
class Subdomain;

struct RestartState {
    double time = 0.0;
    double dt   = 0.0;
    long   step = 0;
    double dPdx = 0.0;
};

class RestartIO {
public:
    RestartIO(const MpiTopology& topo, const Subdomain& sub);

    /// Write all four fields + control state to `dir`. Caller must ensure
    /// the directory exists.
    void write(const std::string& dir,
               const Field<double>& U, const Field<double>& V,
               const Field<double>& W, const Field<double>& P,
               const RestartState& s);

    /// Read all four fields + control state from `dir`. The fields must
    /// already be sized to the local subdomain.
    void read (const std::string& dir,
               Field<double>& U, Field<double>& V,
               Field<double>& W, Field<double>& P,
               RestartState& s);

private:
    void gather_field_(const Field<double>& f, double* global_buf) const;
    void scatter_field_(const double* global_buf, Field<double>& f) const;

    const MpiTopology& topo_;
    const Subdomain&   sub_;
};

} // namespace channel

#endif // CHANNEL_RESTART_IO_HPP
