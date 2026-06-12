// channel/Config.hpp
//
// Parses PARA_INPUT.dat (key = value, # comments) on rank 0
// and broadcasts the parsed values to every rank.
//
// Variable names mirror PaScaL_TCS Fortran namelist groups so
// that input files can be compared 1:1 with the original code:
//
//   group `meshes`            : n1m, n2m, n3m
//   group `MPI_procs`         : np1, np2, np3
//   group `periodic_boundary` : pbc1, pbc2, pbc3
//   group `uniform_mesh`      : uniform1, uniform2, uniform3
//   group `mesh_stretch`      : gamma1, gamma2, gamma3
//   group `aspect_ratio`      : Aspect1, Aspect2, H
//   group `sim_parameter`     : Re_b, MaxCFL
//   group `sim_control`       : dtStart, tStart, Timestepmax
//   group `sim_continue`      : ContinueFilein, ContinueFileout,
//                               dir_cont_filein, dir_cont_fileout,
//                               dir_instantfield, dir_statistics
//   group `channel_forcing`   : forcing_mode, target_bulk_velocity, target_dPdx
//   group `output`            : nmonitor, nstat_start, nstat, nout_stats, nout,
//                               out_stats, out_field

#ifndef CHANNEL_CONFIG_HPP
#define CHANNEL_CONFIG_HPP

#include <mpi.h>
#include <string>

namespace channel {

enum class ForcingMode {
    MASS_FLOW,
    PRESSURE_GRADIENT
};

struct Config {
    // ---- meshes ---------------------------------------------------------
    int n1m = 0, n2m = 0, n3m = 0;
    int n1  = 0, n2  = 0, n3  = 0;   // n*m + 1, derived after parsing

    // ---- MPI_procs ------------------------------------------------------
    int np1 = 1, np2 = 1, np3 = 1;

    // ---- periodic_boundary ---------------------------------------------
    bool pbc1 = true;
    bool pbc2 = true;
    bool pbc3 = false;   // wall-normal default

    // ---- uniform_mesh / mesh_stretch -----------------------------------
    int    uniform1 = 1, uniform2 = 1, uniform3 = 0;
    double gamma1   = 0.0, gamma2 = 0.0, gamma3 = 0.0;

    // ---- aspect_ratio --------------------------------------------------
    // PaScaL_TCS uses (Aspect1, H, Aspect3) where H is wall-to-wall extent.
    // In this code z is wall-normal so Lz = H, Lx = H*Aspect1, Ly = H*Aspect2.
    double H       = 2.0;
    double Aspect1 = 4.0;
    double Aspect2 = 2.0;
    double Lx = 0.0, Ly = 0.0, Lz = 0.0;   // derived

    // ---- sim_parameter -------------------------------------------------
    double Re_b   = 2800.0;   // bulk Reynolds number (=Ub*h/nu, h=Lz/2)
    double MaxCFL = 1.0;

    // ---- sim_control ---------------------------------------------------
    double dtStart      = 1.0e-3;
    double tStart       = 0.0;
    int    Timestepmax  = 10000;

    // ---- sim_continue --------------------------------------------------
    int         ContinueFilein  = 0;
    int         ContinueFileout = 1;
    std::string dir_cont_filein  = "./restart_in/";
    std::string dir_cont_fileout = "./restart_out/";
    std::string dir_instantfield = "./instant/";
    std::string dir_statistics   = "./statistics/";

    // ---- channel_forcing -----------------------------------------------
    ForcingMode forcing_mode         = ForcingMode::MASS_FLOW;
    double      target_bulk_velocity = 1.0;
    double      target_dPdx          = 0.0;

    // ---- initial condition ---------------------------------------------
    // Random perturbation amplitude added to laminar parabolic U profile
    // (as a fraction of Umax). 0 = pure Poiseuille.
    double      pert_amp = 0.01;

    // ---- output timing (v2 convention) ---------------------------------
    //   nmonitor     : print monitor line every N steps
    //   nstat_start  : step at which stats accumulation begins
    //   nstat        : stats accumulation interval (steps), counted from nstat_start
    //   nout_stats   : stats file write interval (steps), counted from nstat_start
    //   nfield_start : step at which instant-field output begins
    //   nout         : instant 3-D field output interval (steps), counted from nfield_start
    //   out_stats    : 1 = write stats files, 0 = skip
    //   out_field    : 1 = write instant field files, 0 = skip
    int nmonitor     = 1;
    int nstat_start  = 0;
    int nstat        = 1;
    int nout_stats   = 1000;
    int nfield_start = 0;
    int nout         = 10000;
    int out_stats    = 1;
    int out_field    = 1;

    // ---- momentum TDMA backend selection -------------------------------
    //   "filtered" (default) — Filtered_TDMA library (truncated-filter v2)
    //   "pascal"             — PaScaL_TDMA library (reduced parallel Thomas)
    std::string tdma_backend = "filtered";

    // ---- Loaders -------------------------------------------------------
    /// Rank 0 reads `path`, parses, and broadcasts; all ranks return populated.
    static Config load(const std::string& path, MPI_Comm comm);

    /// Pretty-print (rank 0 should call).
    void print() const;
};

} // namespace channel

#endif // CHANNEL_CONFIG_HPP
