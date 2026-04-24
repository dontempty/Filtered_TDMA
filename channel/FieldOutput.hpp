// channel/FieldOutput.hpp
//
// Two output modes:
//
//  write_planes()    — 2-D plane snapshots (XY, XZ, YZ) in Tecplot ASCII.
//                      Appended per step → multi-zone time series.
//
//  write_field_3d()  — Full 3-D instantaneous field in Tecplot ASCII.
//                      New file per call: field_%08d.plt
//                      Variables: X Y Z U V W P Q Lambda2
//                      Each rank writes its z-slab in rank order
//                      (rank 0 writes header, then all append sequentially).
//
// Cell-centre velocities (uc, vc, wc) are interpolated from staggered faces.
// Velocity-gradient tensor is computed with second-order central differences;
// for wall-normal z the ghost-cell values from Grid (mirror BC) are used.

#ifndef CHANNEL_FIELD_OUTPUT_HPP
#define CHANNEL_FIELD_OUTPUT_HPP

#include <cstdio>
#include <string>
#include <vector>

#include "Field.hpp"

namespace channel {

class Config;
class MpiTopology;
class Subdomain;
class Grid;

class FieldOutput {
public:
    FieldOutput(const Config&      cfg,
                const MpiTopology& topo,
                const Subdomain&   sub,
                const Grid&        grid);
    ~FieldOutput();

    FieldOutput(const FieldOutput&)            = delete;
    FieldOutput& operator=(const FieldOutput&) = delete;

    // Write XY, XZ, YZ snapshots (appended multi-zone).
    void write_planes(const Field<double>& U,
                      const Field<double>& V,
                      const Field<double>& W,
                      const Field<double>& P,
                      long step, double time);

    // Write full 3-D field with Q-criterion and Lambda2.
    // MPI collective: all ranks participate.
    void write_field_3d(const Field<double>& U,
                        const Field<double>& V,
                        const Field<double>& W,
                        const Field<double>& P,
                        int step);

private:
    // Cell-centre interpolation
    static double uc(const Field<double>& U, int i, int j, int k)
    { return 0.5 * (U(i, j, k) + U(i+1, j, k)); }
    static double vc(const Field<double>& V, int i, int j, int k)
    { return 0.5 * (V(i, j, k) + V(i, j+1, k)); }
    static double wc(const Field<double>& W, int i, int j, int k)
    { return 0.5 * (W(i, j, k) + W(i, j, k+1)); }

    // Middle eigenvalue of symmetric 3×3 (Cardano method) for Lambda2
    static double compute_lambda2_(double M11, double M22, double M33,
                                   double M12, double M13, double M23);

    std::FILE* open_plt_(const std::string& name) const;
    void zone_header_(std::FILE* fp, bool& first,
                      const char* vars,
                      int ni, int nj, int nk,
                      long step, double time) const;

    const Config&      cfg_;
    const MpiTopology& topo_;
    const Subdomain&   sub_;
    const Grid&        grid_;

    // Precomputed cell-centre coordinates (global, 1-indexed).
    std::vector<double> xc_, yc_, zc_;

    int    k_mid_local_   = 0;
    bool   owns_xy_plane_ = false;
    int    j_mid_local_   = 0;
    int    i_mid_local_   = 0;

    bool first_xy_ = true;
    bool first_xz_ = true;
    bool first_yz_ = true;

    std::string rank_sfx_;
};

} // namespace channel

#endif // CHANNEL_FIELD_OUTPUT_HPP
