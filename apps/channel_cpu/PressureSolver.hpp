// channel/PressureSolver.hpp
//
// 2D-FFT (x, y) + distributed 1D-TDMA (z) Poisson solver.
//
// Algorithm (PaScaL_TCS FFT1 adapted for z=wall-normal):
//   [C→I]  alltoall in comm_x  → x-pencil (full x, same y, z split by np1*np3)
//   [FFT x] FFTW r2c on full n1m
//   [I→Y]  alltoallv in comm_y → y-pencil (x-modes split by np2, full y, same z)
//   [FFT y] FFTW c2c on full n2m
//   [TDMA z] distributed PaScaL_TDMA over comm_xz (np1*np3 ranks)
//   [IFFT y+x, Y→I, I→C] reverse
//   [project] velocity correction + accumulate P

#ifndef CHANNEL_PRESSURE_SOLVER_HPP
#define CHANNEL_PRESSURE_SOLVER_HPP

#include <fftw3.h>
#include <complex>
#include <memory>
#include <string>
#include <vector>

#include <mpi.h>
#include "Field.hpp"
#include "pascal_tdma.hpp"

namespace channel {

class MpiTopology;
class Subdomain;
class Grid;
class HaloExchanger;
struct Config;

class PressureSolver {
public:
    PressureSolver(const Config& cfg,
                   const MpiTopology& topo,
                   const Subdomain& sub,
                   const Grid& grid,
                   const HaloExchanger& halo);
    ~PressureSolver();

    PressureSolver(const PressureSolver&)            = delete;
    PressureSolver& operator=(const PressureSolver&) = delete;

    void solve(Field<double>& U, Field<double>& V, Field<double>& W,
               Field<double>& P, double dt);

private:
    void compute_rhs_(const Field<double>& U,
                      const Field<double>& V,
                      const Field<double>& W,
                      double dt);
    void transpose_C_to_I_();
    void transpose_I_to_C_();
    void transpose_I_to_Y_();
    void transpose_Y_to_I_();
    void fft_x_forward_();
    void fft_x_backward_();
    void fft_y_forward_();
    void fft_y_backward_();
    void solve_tdma_z_();
    void project_(Field<double>& U, Field<double>& V,
                  Field<double>& W, Field<double>& P, double dt);

    // ---- topology / sizes ------------------------------------------------
    const MpiTopology* topo_  = nullptr;
    const Subdomain*   sub_   = nullptr;
    const Grid*        grid_  = nullptr;
    const HaloExchanger* halo_ = nullptr;

    int np1_ = 1, np2_ = 1, np3_ = 1;
    int nx_loc_ = 0, ny_loc_ = 0, nz_loc_ = 0;
    int n1m_ = 0, n2m_ = 0, n3m_ = 0;
    int Nxh_ = 0;    // n1m_/2+1 (r2c output)

    // I-pencil: z further split by np1 (PaScaL_TCS FFT1 pattern)
    int n3_I_ = 0;   // nz_loc_ / np1_

    // Y-pencil: x-modes split by np2, full y
    int h1p_Y_me_      = 0;   // this rank's x-freq count
    int ix_start_Y_me_ = 0;   // global x-freq start
    std::vector<int> h1p_Y_;       // [np2_]
    std::vector<int> ix_start_Y_;  // [np2_]
    int n_sys_Y_ = 0;              // h1p_Y_me_ * n2m_

    // comm_xz info (np1*np3 ranks, same iy; key=ix+iz*np1 → sequential z-blocks)
    int rank_xz_ = 0;
    int size_xz_ = 1;

    // ---- wavenumbers / grid -----------------------------------------------
    std::vector<double> kx2_;    // [Nxh_]
    std::vector<double> ky2_;    // [n2m_]
    std::vector<double> dz_, dmz_;
    std::vector<double> dz_g_, dmz_g_;  // global z-metrics, 1-indexed [0..n3m_+1]

    // ---- work buffers -----------------------------------------------------
    std::vector<double>               rhs_C_;  // C-layout  [nx_loc×ny_loc×nz_loc]
    std::vector<double>               rhs_I_;  // I-layout  [n1m×ny_loc×n3_I]
    std::vector<std::complex<double>> hat_I_;  // I-layout  [Nxh×ny_loc×n3_I]
    std::vector<std::complex<double>> hat_Y_;  // Y-layout  [h1p_Y_me×n2m×n3_I]
    std::vector<double>               dp_I_;   // I-layout  [n1m×ny_loc×n3_I]

    // ---- pre-allocated TDMA work arrays [n3_I_ × n_sys_Y_] ---------------
    std::vector<double> tdma_Am_, tdma_Ac_, tdma_Ap_;
    std::vector<double> tdma_Am2_, tdma_Ac2_, tdma_Ap2_;
    std::vector<double> tdma_Be_r_, tdma_Be_c_;

    // ---- pre-allocated transpose buffers (avoid heap churn per solve) ---
    std::vector<double> tx_sbuf_C_, tx_rbuf_C_;   // C↔I transpose buffers
    std::vector<double> tx_sbuf_Y_, tx_rbuf_Y_;   // I↔Y transpose buffers (complex as 2·double)

    // ---- FFTW plans -------------------------------------------------------
    fftw_plan plan_fwd_x_ = nullptr;
    fftw_plan plan_bwd_x_ = nullptr;
    fftw_plan plan_fwd_y_ = nullptr;
    fftw_plan plan_bwd_y_ = nullptr;

    // ---- distributed z-TDMA (PaScaL_TDMA over comm_xz) ------------------
    std::unique_ptr<PaScaLTDMA> tdma_z_;

    // ---- dP field (with halos for projection) ----------------------------
    Field<double> dp_;
};

} // namespace channel

#endif // CHANNEL_PRESSURE_SOLVER_HPP
