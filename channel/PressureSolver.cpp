#include "PressureSolver.hpp"

#include "Config.hpp"
#include "Grid.hpp"
#include "HaloExchanger.hpp"
#include "MpiTopology.hpp"
#include "Subdomain.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace channel {

// ---------------------------------------------------------------------------
// Flat index helpers (0-based interior)
//   C layout: [nx_loc × ny_loc × nz_loc], x-fastest
//   I layout: [n1m    × ny_loc × n3_I  ], x-fastest
//   Y layout: [h1pY   × n2m    × n3_I  ], ix-fastest
// ---------------------------------------------------------------------------
static inline std::size_t idx_C(int i, int j, int k, int nx, int ny)
{ return (std::size_t)i + (std::size_t)nx*((std::size_t)j + (std::size_t)ny*k); }

static inline std::size_t idx_I(int i, int j, int k, int n1m, int ny)
{ return (std::size_t)i + (std::size_t)n1m*((std::size_t)j + (std::size_t)ny*k); }

static inline std::size_t idx_Y(int ix, int j, int k, int nxhY, int n2m)
{ return (std::size_t)ix + (std::size_t)nxhY*((std::size_t)j + (std::size_t)n2m*k); }

// ---------------------------------------------------------------------------
PressureSolver::PressureSolver(const Config& cfg,
                               const MpiTopology& topo,
                               const Subdomain& sub,
                               const Grid& grid,
                               const HaloExchanger& halo)
    : topo_(&topo), sub_(&sub), grid_(&grid), halo_(&halo)
{
    np1_ = topo.dim(0); np2_ = topo.dim(1); np3_ = topo.dim(2);

    nx_loc_ = sub.nx(); ny_loc_ = sub.ny(); nz_loc_ = sub.nz();
    n1m_ = sub.global_n(0); n2m_ = sub.global_n(1); n3m_ = sub.global_n(2);
    Nxh_ = n1m_ / 2 + 1;

    // I-pencil: z further split by np1 (FFT1 pattern)
    if (nz_loc_ % np1_ != 0)
        throw std::runtime_error("[PressureSolver] nz_loc not divisible by np1");
    n3_I_ = nz_loc_ / np1_;

    // Y-pencil: x-freq split by np2 (para_range style)
    h1p_Y_.resize(np2_); ix_start_Y_.resize(np2_);
    {
        int base = Nxh_ / np2_, extra = Nxh_ % np2_;
        for (int r = 0; r < np2_; ++r) {
            h1p_Y_[r]      = base + (r < extra ? 1 : 0);
            ix_start_Y_[r] = r * base + std::min(r, extra);
        }
    }
    int rank_y     = topo.rank_in(1);
    h1p_Y_me_      = h1p_Y_[rank_y];
    ix_start_Y_me_ = ix_start_Y_[rank_y];
    n_sys_Y_       = h1p_Y_me_ * n2m_;

    // comm_xz info
    rank_xz_ = topo.rank_xz();
    size_xz_ = topo.size_xz();

    // Wavenumbers (global grid)
    const double dx = cfg.Lx / n1m_, dy = cfg.Ly / n2m_;
    kx2_.resize(Nxh_);
    ky2_.resize(n2m_);
    for (int i = 0; i < Nxh_; ++i)
        kx2_[i] = 2.0*(1.0 - std::cos(2.0*M_PI*i/n1m_)) / (dx*dx);
    for (int j = 0; j < n2m_; ++j) {
        int jj = (j <= n2m_/2) ? j : j - n2m_;
        ky2_[j] = 2.0*(1.0 - std::cos(2.0*M_PI*jj/n2m_)) / (dy*dy);
    }

    dz_  = grid.dx(2);
    dmz_ = grid.dmx(2);

    // Global z-metrics via Allgather in comm_z
    dz_g_.resize(n3m_ + 2, 0.0);
    dmz_g_.resize(n3m_ + 2, 0.0);
    MPI_Allgather(dz_.data()  + 1, nz_loc_, MPI_DOUBLE,
                  dz_g_.data() + 1, nz_loc_, MPI_DOUBLE, topo.comm(2));
    MPI_Allgather(dmz_.data() + 1, nz_loc_, MPI_DOUBLE,
                  dmz_g_.data()+ 1, nz_loc_, MPI_DOUBLE, topo.comm(2));
    {
        double dmz_last = dmz_[nz_loc_ + 1];
        MPI_Bcast(&dmz_last, 1, MPI_DOUBLE, np3_ - 1, topo.comm(2));
        dmz_g_[n3m_ + 1] = dmz_last;
    }

    // Allocate work buffers
    rhs_C_.assign((std::size_t)nx_loc_ * ny_loc_ * nz_loc_, 0.0);
    rhs_I_.assign((std::size_t)n1m_    * ny_loc_ * n3_I_,   0.0);
    hat_I_.assign((std::size_t)Nxh_    * ny_loc_ * n3_I_,   {0.0,0.0});
    hat_Y_.assign((std::size_t)h1p_Y_me_ * n2m_  * n3_I_,   {0.0,0.0});
    dp_I_ .assign((std::size_t)n1m_    * ny_loc_ * n3_I_,   0.0);
    dp_    = Field<double>(nx_loc_, ny_loc_, nz_loc_);

    // FFTW x-plan: batched ny_loc_*n3_I_ transforms of length n1m_
    {
        int nx[1] = {n1m_};
        plan_fwd_x_ = fftw_plan_many_dft_r2c(
            1, nx, ny_loc_*n3_I_,
            rhs_I_.data(), nullptr, 1, n1m_,
            reinterpret_cast<fftw_complex*>(hat_I_.data()), nullptr, 1, Nxh_,
            FFTW_ESTIMATE);
        plan_bwd_x_ = fftw_plan_many_dft_c2r(
            1, nx, ny_loc_*n3_I_,
            reinterpret_cast<fftw_complex*>(hat_I_.data()), nullptr, 1, Nxh_,
            dp_I_.data(), nullptr, 1, n1m_,
            FFTW_ESTIMATE);
    }

    // FFTW y-plan: for each k-slice, h1p_Y_me_ transforms of length n2m_
    // hat_Y_[ix + h1p_Y_me_*(j + n2m_*k)]: stride=h1p_Y_me_, dist=1
    if (h1p_Y_me_ > 0) {
        int ny[1] = {n2m_};
        auto* p = reinterpret_cast<fftw_complex*>(hat_Y_.data());
        plan_fwd_y_ = fftw_plan_many_dft(
            1, ny, h1p_Y_me_,
            p, nullptr, h1p_Y_me_, 1,
            p, nullptr, h1p_Y_me_, 1,
            FFTW_FORWARD, FFTW_ESTIMATE);
        plan_bwd_y_ = fftw_plan_many_dft(
            1, ny, h1p_Y_me_,
            p, nullptr, h1p_Y_me_, 1,
            p, nullptr, h1p_Y_me_, 1,
            FFTW_BACKWARD, FFTW_ESTIMATE);
    }

    // Distributed z-TDMA (PaScaL_TDMA over comm_xz)
    if (n_sys_Y_ > 0) {
        tdma_z_ = std::make_unique<PaScaLTDMAMany>(
            n_sys_Y_, rank_xz_, size_xz_, topo.comm_xz());
        std::size_t tsz = (std::size_t)n3_I_ * n_sys_Y_;
        tdma_Am_.resize(tsz); tdma_Ac_.resize(tsz); tdma_Ap_.resize(tsz);
        tdma_Am2_.resize(tsz); tdma_Ac2_.resize(tsz); tdma_Ap2_.resize(tsz);
        tdma_Be_r_.resize(tsz); tdma_Be_c_.resize(tsz);
    }

    // Pre-allocate transpose buffers (reused every solve)
    {
        const std::size_t blk_C = (std::size_t)nx_loc_ * ny_loc_ * n3_I_;
        tx_sbuf_C_.resize(np1_ * blk_C);
        tx_rbuf_C_.resize(np1_ * blk_C);
        // I↔Y forward sends Nxh*yn3*2, backward sends np2*(h1p_me*yn3*2). Take max.
        const std::size_t yn3     = (std::size_t)ny_loc_ * n3_I_;
        const std::size_t nfwd    = (std::size_t)Nxh_        * yn3 * 2;
        const std::size_t nbwd    = (std::size_t)np2_ * (std::size_t)h1p_Y_me_ * yn3 * 2;
        const std::size_t n_cplx_Y = std::max(nfwd, nbwd);
        tx_sbuf_Y_.resize(n_cplx_Y);
        tx_rbuf_Y_.resize(n_cplx_Y);
    }
}

PressureSolver::~PressureSolver()
{
    if (plan_fwd_x_) fftw_destroy_plan(plan_fwd_x_);
    if (plan_bwd_x_) fftw_destroy_plan(plan_bwd_x_);
    if (plan_fwd_y_) fftw_destroy_plan(plan_fwd_y_);
    if (plan_bwd_y_) fftw_destroy_plan(plan_bwd_y_);
}

// ---------------------------------------------------------------------------
// RHS: (1/dt) div(u*) in C-layout
// ---------------------------------------------------------------------------
void PressureSolver::compute_rhs_(const Field<double>& U,
                                  const Field<double>& V,
                                  const Field<double>& W,
                                  double dt)
{
    const auto& dx = grid_->dx(0);
    const auto& dy = grid_->dx(1);
    const auto& dz = grid_->dx(2);
    const double inv_dt = 1.0 / dt;

    for (int k = 1; k <= nz_loc_; ++k)
        for (int j = 1; j <= ny_loc_; ++j)
            for (int i = 1; i <= nx_loc_; ++i) {
                double d = (U(i+1,j,k)-U(i,j,k))/dx[i]
                         + (V(i,j+1,k)-V(i,j,k))/dy[j]
                         + (W(i,j,k+1)-W(i,j,k))/dz[k];
                rhs_C_[idx_C(i-1,j-1,k-1,nx_loc_,ny_loc_)] = d * inv_dt;
            }
}

// ---------------------------------------------------------------------------
// C → I  (MPI_Alltoall in comm_x, np1_ ranks)
// New: z further split by np1 → n3_I_ = nz_loc_/np1_
// rhs_C_[nx_loc×ny_loc×nz_loc] → rhs_I_[n1m×ny_loc×n3_I]
// ---------------------------------------------------------------------------
void PressureSolver::transpose_C_to_I_()
{
    const int blk = nx_loc_ * ny_loc_ * n3_I_;
    double* sbuf = tx_sbuf_C_.data();
    double* rbuf = tx_rbuf_C_.data();

    // Pack: send to rank s the z-rows [s*n3_I_, (s+1)*n3_I_)
    for (int s = 0; s < np1_; ++s) {
        double* sb = sbuf + (std::size_t)s * blk;
        for (int i = 0; i < nx_loc_; ++i)
            for (int j = 0; j < ny_loc_; ++j)
                for (int k = 0; k < n3_I_; ++k)
                    sb[i*ny_loc_*n3_I_ + j*n3_I_ + k] =
                        rhs_C_[idx_C(i, j, s*n3_I_+k, nx_loc_, ny_loc_)];
    }

    MPI_Alltoall(sbuf, blk, MPI_DOUBLE,
                 rbuf, blk, MPI_DOUBLE,
                 topo_->comm_x());

    // Unpack: from rank s → x-block [s*nx_loc_, (s+1)*nx_loc_)
    for (int s = 0; s < np1_; ++s) {
        const double* rb = rbuf + (std::size_t)s * blk;
        for (int i = 0; i < nx_loc_; ++i)
            for (int j = 0; j < ny_loc_; ++j)
                for (int k = 0; k < n3_I_; ++k)
                    rhs_I_[idx_I(s*nx_loc_+i, j, k, n1m_, ny_loc_)] =
                        rb[i*ny_loc_*n3_I_ + j*n3_I_ + k];
    }
}

// ---------------------------------------------------------------------------
// I → C  (reverse of C→I)
// dp_I_[n1m×ny_loc×n3_I] → dp_ (Field interior, accumulated with +=)
// ---------------------------------------------------------------------------
void PressureSolver::transpose_I_to_C_()
{
    const int blk = nx_loc_ * ny_loc_ * n3_I_;
    double* sbuf = tx_sbuf_C_.data();
    double* rbuf = tx_rbuf_C_.data();

    // Pack: from rank s x-block [s*nx_loc_]
    for (int s = 0; s < np1_; ++s) {
        double* sb = sbuf + (std::size_t)s * blk;
        for (int i = 0; i < nx_loc_; ++i)
            for (int j = 0; j < ny_loc_; ++j)
                for (int k = 0; k < n3_I_; ++k)
                    sb[i*ny_loc_*n3_I_ + j*n3_I_ + k] =
                        dp_I_[idx_I(s*nx_loc_+i, j, k, n1m_, ny_loc_)];
    }

    MPI_Alltoall(sbuf, blk, MPI_DOUBLE,
                 rbuf, blk, MPI_DOUBLE,
                 topo_->comm_x());

    // Unpack: to z-rows [s*n3_I_, (s+1)*n3_I_)
    for (int s = 0; s < np1_; ++s) {
        const double* rb = rbuf + (std::size_t)s * blk;
        for (int i = 0; i < nx_loc_; ++i)
            for (int j = 0; j < ny_loc_; ++j)
                for (int k = 0; k < n3_I_; ++k)
                    dp_(i+1, j+1, s*n3_I_+k+1) +=
                        rb[i*ny_loc_*n3_I_ + j*n3_I_ + k];
    }
}

// ---------------------------------------------------------------------------
// I → Y  (MPI_Alltoallv in comm_y, np2_ ranks)
// hat_I_[Nxh×ny_loc×n3_I] → hat_Y_[h1p_Y_me×n2m×n3_I]
// Assembles full y, splits x-modes by np2
// ---------------------------------------------------------------------------
void PressureSolver::transpose_I_to_Y_()
{
    const int yn3 = ny_loc_ * n3_I_;

    std::vector<int> scnts(np2_), sdsp(np2_),
                     rcnts(np2_), rdsp(np2_);
    {
        int off = 0;
        for (int r = 0; r < np2_; ++r) {
            scnts[r] = h1p_Y_[r] * yn3 * 2;  // *2: complex → 2 doubles
            sdsp[r]  = off;
            off += scnts[r];
        }
    }
    const int rcnt_each = h1p_Y_me_ * yn3 * 2;
    for (int r = 0; r < np2_; ++r) {
        rcnts[r] = rcnt_each;
        rdsp[r]  = r * rcnt_each;
    }

    double* sbuf = tx_sbuf_Y_.data();
    double* rbuf = tx_rbuf_Y_.data();

    // Pack: to rank r, send x-modes [ix_start_Y_[r]..+h1p_Y_[r]-1], all j, all k
    {
        int off = 0;
        for (int r = 0; r < np2_; ++r) {
            double* sb = sbuf + off;
            int ix0 = ix_start_Y_[r], nxr = h1p_Y_[r];
            for (int ixl = 0; ixl < nxr; ++ixl)
                for (int j = 0; j < ny_loc_; ++j)
                    for (int k = 0; k < n3_I_; ++k) {
                        auto c = hat_I_[idx_I(ix0+ixl, j, k, Nxh_, ny_loc_)];
                        int p = (ixl*yn3 + j*n3_I_ + k) * 2;
                        sb[p] = c.real(); sb[p+1] = c.imag();
                    }
            off += scnts[r];
        }
    }

    MPI_Alltoallv(sbuf, scnts.data(), sdsp.data(), MPI_DOUBLE,
                  rbuf, rcnts.data(), rdsp.data(), MPI_DOUBLE,
                  topo_->comm_y());

    // Unpack: from rank r → y-block [r*ny_loc_, (r+1)*ny_loc_)
    for (int r = 0; r < np2_; ++r) {
        const double* rb = rbuf + (std::size_t)r * rcnt_each;
        int j_base = r * ny_loc_;
        for (int ixl = 0; ixl < h1p_Y_me_; ++ixl)
            for (int jl = 0; jl < ny_loc_; ++jl)
                for (int k = 0; k < n3_I_; ++k) {
                    int p = (ixl*yn3 + jl*n3_I_ + k) * 2;
                    hat_Y_[idx_Y(ixl, j_base+jl, k, h1p_Y_me_, n2m_)] =
                        {rb[p], rb[p+1]};
                }
    }
}

// ---------------------------------------------------------------------------
// Y → I  (reverse of I→Y)
// ---------------------------------------------------------------------------
void PressureSolver::transpose_Y_to_I_()
{
    const int yn3 = ny_loc_ * n3_I_;
    const int scnt_each = h1p_Y_me_ * yn3 * 2;

    std::vector<int> scnts(np2_), sdsp(np2_),
                     rcnts(np2_), rdsp(np2_);
    for (int r = 0; r < np2_; ++r) {
        scnts[r] = scnt_each;
        sdsp[r]  = r * scnt_each;
        rcnts[r] = h1p_Y_[r] * yn3 * 2;
    }
    {
        int off = 0;
        for (int r = 0; r < np2_; ++r) { rdsp[r] = off; off += rcnts[r]; }
    }

    double* sbuf = tx_sbuf_Y_.data();
    double* rbuf = tx_rbuf_Y_.data();

    // Pack: to rank r, send y-block [r*ny_loc_, (r+1)*ny_loc_)
    for (int r = 0; r < np2_; ++r) {
        double* sb = sbuf + (std::size_t)r * scnt_each;
        int j_base = r * ny_loc_;
        for (int ixl = 0; ixl < h1p_Y_me_; ++ixl)
            for (int jl = 0; jl < ny_loc_; ++jl)
                for (int k = 0; k < n3_I_; ++k) {
                    auto c = hat_Y_[idx_Y(ixl, j_base+jl, k, h1p_Y_me_, n2m_)];
                    int p = (ixl*yn3 + jl*n3_I_ + k) * 2;
                    sb[p] = c.real(); sb[p+1] = c.imag();
                }
    }

    MPI_Alltoallv(sbuf, scnts.data(), sdsp.data(), MPI_DOUBLE,
                  rbuf, rcnts.data(), rdsp.data(), MPI_DOUBLE,
                  topo_->comm_y());

    // Unpack: from rank r → x-modes [ix_start_Y_[r]..+h1p_Y_[r]-1]
    int off = 0;
    for (int r = 0; r < np2_; ++r) {
        const double* rb = rbuf + off;
        int ix0 = ix_start_Y_[r], nxr = h1p_Y_[r];
        for (int ixl = 0; ixl < nxr; ++ixl)
            for (int j = 0; j < ny_loc_; ++j)
                for (int k = 0; k < n3_I_; ++k) {
                    int p = (ixl*yn3 + j*n3_I_ + k) * 2;
                    hat_I_[idx_I(ix0+ixl, j, k, Nxh_, ny_loc_)] = {rb[p], rb[p+1]};
                }
        off += rcnts[r];
    }
}

// ---------------------------------------------------------------------------
// FFT wrappers
// ---------------------------------------------------------------------------
void PressureSolver::fft_x_forward_()  { if (plan_fwd_x_) fftw_execute(plan_fwd_x_); }
void PressureSolver::fft_x_backward_() { if (plan_bwd_x_) fftw_execute(plan_bwd_x_); }

void PressureSolver::fft_y_forward_()
{
    if (!plan_fwd_y_) return;
    auto* p = reinterpret_cast<fftw_complex*>(hat_Y_.data());
    const std::size_t stride = (std::size_t)h1p_Y_me_ * n2m_;
    for (int k = 0; k < n3_I_; ++k)
        fftw_execute_dft(plan_fwd_y_, p + k*stride, p + k*stride);
}

void PressureSolver::fft_y_backward_()
{
    if (!plan_bwd_y_) return;
    auto* p = reinterpret_cast<fftw_complex*>(hat_Y_.data());
    const std::size_t stride = (std::size_t)h1p_Y_me_ * n2m_;
    for (int k = 0; k < n3_I_; ++k)
        fftw_execute_dft(plan_bwd_y_, p + k*stride, p + k*stride);
}

// ---------------------------------------------------------------------------
// z-TDMA: distributed PaScaL_TDMA over comm_xz (np1*np3 ranks)
// Y-pencil: hat_Y_[h1p_Y_me_ × n2m_ × n3_I_]
// System index: s = ixl + h1p_Y_me_ * j  (ixl: local x-mode, j: global y-mode)
// k_global = rank_xz_ * n3_I_ + k  (0-based wall-normal index)
// ---------------------------------------------------------------------------
void PressureSolver::solve_tdma_z_()
{
    if (!tdma_z_ || n_sys_Y_ == 0 || n3_I_ == 0) return;

    double* Am   = tdma_Am_.data();
    double* Ac   = tdma_Ac_.data();
    double* Ap   = tdma_Ap_.data();
    double* Am2  = tdma_Am2_.data();
    double* Ac2  = tdma_Ac2_.data();
    double* Ap2  = tdma_Ap2_.data();
    double* Be_r = tdma_Be_r_.data();
    double* Be_c = tdma_Be_c_.data();

    for (int k = 0; k < n3_I_; ++k) {
        int k_g = rank_xz_ * n3_I_ + k;  // 0-based global z-index
        const double dz  = dz_g_ [k_g + 1];
        const double dmz = dmz_g_[k_g + 1];
        const double dmzp= dmz_g_[k_g + 2];

        const bool lower_wall = (rank_xz_ == 0          && k == 0);
        const bool upper_wall = (rank_xz_ == size_xz_-1 && k == n3_I_-1);

        const double a_val = lower_wall ? 0.0 : 1.0/(dz*dmz);
        const double c_val = upper_wall ? 0.0 : 1.0/(dz*dmzp);

        for (int j = 0; j < n2m_; ++j) {
            for (int ixl = 0; ixl < h1p_Y_me_; ++ixl) {
                int s = ixl + h1p_Y_me_ * j;
                const double lambda = kx2_[ix_start_Y_me_ + ixl] + ky2_[j];
                const std::size_t idx = (std::size_t)k * n_sys_Y_ + s;
                Am[idx] = Am2[idx] = a_val;
                Ap[idx] = Ap2[idx] = c_val;
                Ac[idx] = Ac2[idx] = -(a_val + c_val) - lambda;
                auto cv = hat_Y_[idx_Y(ixl, j, k, h1p_Y_me_, n2m_)];
                Be_r[idx] = cv.real();
                Be_c[idx] = cv.imag();
            }
        }
    }

    // Zero-mode: kx=0, ky=0 at rank_xz_==0
    if (rank_xz_ == 0 && ix_start_Y_me_ == 0 && h1p_Y_me_ > 0) {
        Am[0] = Am2[0] = 0.0; Ac[0] = Ac2[0] = 1.0; Ap[0] = Ap2[0] = 0.0;
        Be_r[0] = 0.0; Be_c[0] = 0.0;
    }

    // solve() modifies A, C in-place → use Am/Ac/Ap for real, Am2/Ac2/Ap2 for imag
    tdma_z_->solve(Am,  Ac,  Ap,  Be_r, n_sys_Y_, n3_I_);
    tdma_z_->solve(Am2, Ac2, Ap2, Be_c, n_sys_Y_, n3_I_);

    // Write solution back to hat_Y_
    for (int k = 0; k < n3_I_; ++k)
        for (int j = 0; j < n2m_; ++j)
            for (int ixl = 0; ixl < h1p_Y_me_; ++ixl) {
                int s = ixl + h1p_Y_me_ * j;
                const std::size_t idx = (std::size_t)k * n_sys_Y_ + s;
                hat_Y_[idx_Y(ixl, j, k, h1p_Y_me_, n2m_)] = {Be_r[idx], Be_c[idx]};
            }
}

// ---------------------------------------------------------------------------
// Projection: u^{n+1} = u* - dt·∇(δP),  P^{n+1} = P^n + δP
// ---------------------------------------------------------------------------
void PressureSolver::project_(Field<double>& U, Field<double>& V,
                               Field<double>& W, Field<double>& P, double dt)
{
    const auto& dmx = grid_->dmx(0);
    const auto& dmy = grid_->dmx(1);
    const auto& dmz = grid_->dmx(2);

    for (int k = 1; k <= nz_loc_; ++k)
        for (int j = 1; j <= ny_loc_; ++j)
            for (int i = 1; i <= nx_loc_; ++i) {
                P(i,j,k) += dp_(i,j,k);
                U(i,j,k) -= dt * (dp_(i,j,k) - dp_(i-1,j,k)) / dmx[i];
                V(i,j,k) -= dt * (dp_(i,j,k) - dp_(i,j-1,k)) / dmy[j];
                W(i,j,k) -= dt * (dp_(i,j,k) - dp_(i,j,k-1)) / dmz[k];
            }
}

// ---------------------------------------------------------------------------
void PressureSolver::solve(Field<double>& U, Field<double>& V, Field<double>& W,
                           Field<double>& P, double dt)
{
    compute_rhs_(U, V, W, dt);

    // Forward: C→I, FFT(x), I→Y, FFT(y)
    transpose_C_to_I_();
    fft_x_forward_();
    transpose_I_to_Y_();
    fft_y_forward_();

    // Tridiagonal solve in z (distributed PaScaL_TDMA over comm_xz)
    solve_tdma_z_();

    // Backward: IFFT(y), Y→I, IFFT(x)
    fft_y_backward_();
    transpose_Y_to_I_();
    fft_x_backward_();

    // Normalise (FFTW un-normalised) and copy into dp_ field
    const double scale = 1.0 / (static_cast<double>(n1m_) * n2m_);
    for (std::size_t p = 0; p < dp_I_.size(); ++p) dp_I_[p] *= scale;

    dp_ = Field<double>(nx_loc_, ny_loc_, nz_loc_);
    transpose_I_to_C_();

    // Neumann BC for dp_ at z walls (ghost = first interior)
    if (topo_->coord(2) == 0)
        for (int j = 1; j <= ny_loc_; ++j)
            for (int i = 1; i <= nx_loc_; ++i)
                dp_(i,j,0) = dp_(i,j,1);
    if (topo_->coord(2) == topo_->dim(2)-1)
        for (int j = 1; j <= ny_loc_; ++j)
            for (int i = 1; i <= nx_loc_; ++i)
                dp_(i,j,nz_loc_+1) = dp_(i,j,nz_loc_);

    // Periodic halo exchange for x, y ghost cells of dp_
    halo_->exchange(dp_);

    project_(U, V, W, P, dt);
}

} // namespace channel
