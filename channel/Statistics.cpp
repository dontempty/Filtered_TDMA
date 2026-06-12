#include "Statistics.hpp"

#include "Grid.hpp"
#include "MpiTopology.hpp"
#include "Subdomain.hpp"

#include <cmath>
#include <cstdio>
#include <mpi.h>

namespace channel {

// ============================================================
//  Statistics — HYBRID convention:
//
//  Cell-center stats (zc[k]):  U_m, V_m, Wc_m, U2_m, V2_m, Wc2_m, P_m
//    - Reverted to OLD-style formula (no z-halo dependence).
//    - u_rms, v_rms, w_rms, P_mean are read from these.
//
//  Corner stats (zf[k]):       Ug_m, Wg_m, UWg_m
//    - MPM-STD style corner interpolation.
//    - For uw cross stat: avoids the PaScaL rank-boundary bias
//      that affects Wc when interpolating across rank ghost.
//    - Output uw at zc[k] = interpolation of corner uw at zf[k] and zf[k+1].
// ============================================================

Statistics::Statistics(const MpiTopology& topo,
                       const Subdomain&   sub,
                       const Grid&        grid)
    : topo_(topo), sub_(sub), grid_(grid)
{
    nz_global_ = sub_.global_n(2);
    nz_local_  = sub_.nz();
    kstart_    = sub_.ista(2) - 1;   // 0-based global offset

    U_m_  .assign(nz_local_, 0.0);  U2_m_ .assign(nz_local_, 0.0);
    V_m_  .assign(nz_local_, 0.0);  V2_m_ .assign(nz_local_, 0.0);
    Wc_m_ .assign(nz_local_, 0.0);  Wc2_m_.assign(nz_local_, 0.0);
    P_m_  .assign(nz_local_, 0.0);

    Ug_m_ .assign(nz_local_, 0.0);
    Wg_m_ .assign(nz_local_, 0.0);
    UWg_m_.assign(nz_local_, 0.0);

    // Gather LOCAL dz arrays across z-ranks to build a global dz array
    const auto& dz_local = grid_.dx(2);
    int nz_local = sub_.nz();

    std::vector<double> dz_local_flat(nz_local);
    for (int k = 0; k < nz_local; ++k)
        dz_local_flat[k] = dz_local[k + 1];

    std::vector<double> dz_global(nz_global_);
    MPI_Allgather(dz_local_flat.data(), nz_local, MPI_DOUBLE,
                  dz_global.data(),    nz_local, MPI_DOUBLE,
                  topo_.comm_z());

    // Cell-center z positions
    zc_global_.resize(nz_global_);
    double cum = 0.0;
    for (int k = 0; k < nz_global_; ++k) {
        zc_global_[k] = cum + dz_global[k] * 0.5;
        cum          += dz_global[k];
    }

    // z-face positions (zf[k] = lower face of global cell k)
    z_face_global_.resize(nz_global_);
    cum = 0.0;
    for (int k = 0; k < nz_global_; ++k) {
        z_face_global_[k] = cum;
        cum              += dz_global[k];
    }
}

void Statistics::reset()
{
    n_ = 0;
    std::fill(U_m_  .begin(), U_m_  .end(), 0.0);
    std::fill(U2_m_ .begin(), U2_m_ .end(), 0.0);
    std::fill(V_m_  .begin(), V_m_  .end(), 0.0);
    std::fill(V2_m_ .begin(), V2_m_ .end(), 0.0);
    std::fill(Wc_m_ .begin(), Wc_m_ .end(), 0.0);
    std::fill(Wc2_m_.begin(), Wc2_m_.end(), 0.0);
    std::fill(P_m_  .begin(), P_m_  .end(), 0.0);
    std::fill(Ug_m_ .begin(), Ug_m_ .end(), 0.0);
    std::fill(Wg_m_ .begin(), Wg_m_ .end(), 0.0);
    std::fill(UWg_m_.begin(), UWg_m_.end(), 0.0);
}

void Statistics::gather_to_global(const std::vector<double>& local,
                                   std::vector<double>& global) const
{
    global.assign(nz_global_, 0.0);
    std::vector<double> tmp(nz_global_, 0.0);
    for (int kl = 0; kl < nz_local_; ++kl)
        tmp[kstart_ + kl] = local[kl];
    MPI_Allreduce(tmp.data(), global.data(),
                  nz_global_, MPI_DOUBLE, MPI_SUM, topo_.cart());
}

// Gather many fields in a single Allreduce (coalesced).
static void gather_many(const std::vector<const std::vector<double>*>& locals,
                        std::vector<std::vector<double>*>& globals,
                        int nz_global, int nz_local, int kstart,
                        MPI_Comm comm)
{
    const int N = static_cast<int>(locals.size());
    std::vector<double> tmp(static_cast<std::size_t>(N) * nz_global, 0.0);
    for (int f = 0; f < N; ++f) {
        double* row = tmp.data() + static_cast<std::size_t>(f) * nz_global;
        const auto& L = *locals[f];
        for (int kl = 0; kl < nz_local; ++kl) row[kstart + kl] = L[kl];
    }
    std::vector<double> out(tmp.size(), 0.0);
    MPI_Allreduce(tmp.data(), out.data(),
                  static_cast<int>(tmp.size()), MPI_DOUBLE, MPI_SUM, comm);
    for (int f = 0; f < N; ++f) {
        auto& G = *globals[f];
        G.assign(nz_global, 0.0);
        const double* row = out.data() + static_cast<std::size_t>(f) * nz_global;
        for (int k = 0; k < nz_global; ++k) G[k] = row[k];
    }
}

// ============================================================
//  accumulate — both cell-center and corner stats in one loop
// ============================================================
void Statistics::accumulate(const Field<double>& U,
                             const Field<double>& V,
                             const Field<double>& W,
                             const Field<double>& P)
{
    const int nx   = sub_.nx(), ny = sub_.ny();
    const int nx_g = sub_.global_n(0);
    const int ny_g = sub_.global_n(1);
    const double inv_NxNy = 1.0 / static_cast<double>(nx_g * ny_g);

    const auto& dx = grid_.dx(0);
    const auto& dy = grid_.dx(1);
    const auto& dz = grid_.dx(2);

    ++n_;
    const double inv_n = 1.0 / static_cast<double>(n_);

    for (int kl = 1; kl <= nz_local_; ++kl) {
        // ---------- Cell-center accumulators (OLD style) ----------
        double su=0, su2=0, sv=0, sv2=0;
        double swc=0, swc2=0, sp=0;

        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i) {
                const double uc  = 0.5 * (U(i, j, kl) + U(i + 1, j, kl));
                const double vc  = 0.5 * (V(i, j, kl) + V(i, j + 1, kl));
                const double wci = 0.5 * (W(i, j, kl) + W(i, j, kl + 1));
                const double pi  = P(i, j, kl);
                su   += uc;   su2  += uc  * uc;
                sv   += vc;   sv2  += vc  * vc;
                swc  += wci;  swc2 += wci * wci;
                sp   += pi;
            }

        // ---------- Corner accumulators (NEW MPM-STD style) ----------
        const int km = kl - 1;
        const double w_kl = dz[km];
        const double w_km = dz[kl];
        const double zsum = dz[kl] + dz[km];

        double sug=0, swg=0, sugwg=0;

        for (int j = 1; j <= ny; ++j) {
            const int jm = j - 1;
            const double yw_j  = dy[jm];
            const double yw_jm = dy[j];
            const double ysum  = dy[j] + dy[jm];

            for (int i = 1; i <= nx; ++i) {
                const int im = i - 1;
                const double xw_i  = dx[im];
                const double xw_im = dx[i];
                const double xsum  = dx[i] + dx[im];

                // Ug at corner (i-1/2, j-1/2, kl-1/2): U at x-face (i-1/2, *, *)
                const double Uy_km = (U(i,j,km)*yw_j + U(i,jm,km)*yw_jm) / ysum;
                const double Uy_kl = (U(i,j,kl)*yw_j + U(i,jm,kl)*yw_jm) / ysum;
                const double Ug    = (Uy_kl*w_kl + Uy_km*w_km) / zsum;

                // Wg at corner: W at z-face (*, *, kl-1/2) — only x,y-interp
                const double Wx_jm = (W(i,jm,kl)*xw_i + W(im,jm,kl)*xw_im) / xsum;
                const double Wx_j  = (W(i,j ,kl)*xw_i + W(im,j ,kl)*xw_im) / xsum;
                const double Wg    = (Wx_j*yw_j + Wx_jm*yw_jm) / ysum;

                sug   += Ug;
                swg   += Wg;
                sugwg += Ug * Wg;
            }
        }

        // ---------- Welford updates ----------
        const int    kl0  = kl - 1;
        const double um   = su    * inv_NxNy;
        const double u2m  = su2   * inv_NxNy;
        const double vm   = sv    * inv_NxNy;
        const double v2m  = sv2   * inv_NxNy;
        const double wcm  = swc   * inv_NxNy;
        const double wc2m = swc2  * inv_NxNy;
        const double pm   = sp    * inv_NxNy;
        const double ugm  = sug   * inv_NxNy;
        const double wgm  = swg   * inv_NxNy;
        const double ugwgm= sugwg * inv_NxNy;

        U_m_  [kl0] += (um    - U_m_  [kl0]) * inv_n;
        U2_m_ [kl0] += (u2m   - U2_m_ [kl0]) * inv_n;
        V_m_  [kl0] += (vm    - V_m_  [kl0]) * inv_n;
        V2_m_ [kl0] += (v2m   - V2_m_ [kl0]) * inv_n;
        Wc_m_ [kl0] += (wcm   - Wc_m_ [kl0]) * inv_n;
        Wc2_m_[kl0] += (wc2m  - Wc2_m_[kl0]) * inv_n;
        P_m_  [kl0] += (pm    - P_m_  [kl0]) * inv_n;
        Ug_m_ [kl0] += (ugm   - Ug_m_ [kl0]) * inv_n;
        Wg_m_ [kl0] += (wgm   - Wg_m_ [kl0]) * inv_n;
        UWg_m_[kl0] += (ugwgm - UWg_m_[kl0]) * inv_n;
    }
}

// ============================================================
//  write — gather all stats, output at zc[k]
//    u_rms, v_rms, w_rms, P_mean : cell-center (clean)
//    uw_stress                    : corner-based, interpolated to zc[k]
// ============================================================
void Statistics::write(const std::string& path, int step, double Re_b,
                        bool reset_after)
{
    std::vector<double> U_g, U2_g, V_g, V2_g, Wc_g, Wc2_g, P_g;
    std::vector<double> Ug_g, Wg_g, UWg_g;
    {
        std::vector<const std::vector<double>*> locals = {
            &U_m_, &U2_m_, &V_m_, &V2_m_, &Wc_m_, &Wc2_m_, &P_m_,
            &Ug_m_, &Wg_m_, &UWg_m_
        };
        std::vector<std::vector<double>*> globals = {
            &U_g, &U2_g, &V_g, &V2_g, &Wc_g, &Wc2_g, &P_g,
            &Ug_g, &Wg_g, &UWg_g
        };
        gather_many(locals, globals, nz_global_, nz_local_, kstart_, topo_.cart());
    }

    const long n_saved = n_;
    if (reset_after) reset();

    if (topo_.rank() != 0 || n_saved == 0) return;

    // u_tau from cell-center first cell (OLD style).
    const double nu     = 1.0 / Re_b;
    const double tau_w  = nu * std::abs(U_g[0]) / zc_global_[0];
    const double u_tau  = std::sqrt(std::max(tau_w, 0.0));
    const double inv_nu = Re_b;

    FILE* fp = fopen(path.c_str(), "w");
    if (!fp) {
        fprintf(stderr, "[Statistics::write] Cannot open %s\n", path.c_str());
        return;
    }

    fprintf(fp, "TITLE = \"Channel Flow Statistics (step=%d, n_samples=%ld)\"\n",
            step, n_saved);
    fprintf(fp, "VARIABLES = \"Z\" \"Z_plus\""
                " \"U_mean\" \"W_mean\""
                " \"u_rms\" \"v_rms\" \"w_rms\""
                " \"uw_stress\" \"P_mean\"\n");
    fprintf(fp, "ZONE T=\"Stats\", I=%d, J=1, K=1, DATAPACKING=POINT\n",
            nz_global_);

    for (int k = 0; k < nz_global_; ++k) {
        const double z     = zc_global_[k];
        const double zp    = z * u_tau * inv_nu;

        // Cell-center variance/means (clean)
        const double u_rms = std::sqrt(std::max(U2_g[k]  - U_g[k]  * U_g[k],  0.0));
        const double v_rms = std::sqrt(std::max(V2_g[k]  - V_g[k]  * V_g[k],  0.0));
        const double w_rms = std::sqrt(std::max(Wc2_g[k] - Wc_g[k] * Wc_g[k], 0.0));

        // uw at zc[k]: average of corner uw at zf[k] (= UWg row k) and
        // zf[k+1] (= UWg row k+1). For k=n3m-1, only zf[k] is available.
        const double uw_face_k = UWg_g[k] - Ug_g[k] * Wg_g[k];
        double uw;
        if (k + 1 < nz_global_) {
            const double uw_face_kp1 = UWg_g[k+1] - Ug_g[k+1] * Wg_g[k+1];
            uw = 0.5 * (uw_face_k + uw_face_kp1);
        } else {
            uw = uw_face_k;
        }

        fprintf(fp, "%.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e\n",
                z, zp,
                U_g[k], Wc_g[k],
                u_rms, v_rms, w_rms,
                uw, P_g[k]);
    }
    fclose(fp);
}

} // namespace channel
