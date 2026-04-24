#include "Statistics.hpp"

#include "Grid.hpp"
#include "MpiTopology.hpp"
#include "Subdomain.hpp"

#include <cmath>
#include <cstdio>
#include <mpi.h>

namespace channel {

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
    UWc_m_.assign(nz_local_, 0.0);  P_m_  .assign(nz_local_, 0.0);

    // Precompute global cell-centre z-coordinates (0-indexed)
    const auto& dz = grid_.dx(2);
    zc_global_.resize(nz_global_);
    double cum = 0.0;
    for (int k = 0; k < nz_global_; ++k) {
        zc_global_[k] = cum + dz[k + 1] * 0.5;   // dz is 1-indexed
        cum           += dz[k + 1];
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
    std::fill(UWc_m_.begin(), UWc_m_.end(), 0.0);
    std::fill(P_m_  .begin(), P_m_  .end(), 0.0);
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

// Gather all 8 stat fields in a single Allreduce (coalesced).
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
//  accumulate — zero MPI, local Welford incremental mean
//
//  For each z-level: xy-plane mean of u, u², v, v², wc, wc², u·wc, p.
//  wc = 0.5*(W_bottom_face + W_top_face): z-face → cell-centre.
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

    ++n_;
    const double inv_n = 1.0 / static_cast<double>(n_);

    for (int kl = 1; kl <= nz_local_; ++kl) {
        double su=0, su2=0, sv=0, sv2=0;
        double swc=0, swc2=0, suwc=0, sp=0;

        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i) {
                const double uc  = 0.5 * (U(i, j, kl) + U(i + 1, j, kl));
                const double vc  = 0.5 * (V(i, j, kl) + V(i, j + 1, kl));
                const double wci = 0.5 * (W(i, j, kl) + W(i, j, kl + 1));
                const double pi  = P(i, j, kl);
                su   += uc;   su2  += uc  * uc;
                sv   += vc;   sv2  += vc  * vc;
                swc  += wci;  swc2 += wci * wci;
                suwc += uc * wci;
                sp   += pi;
            }

        const int    kl0  = kl - 1;   // 0-indexed local
        const double um   = su   * inv_NxNy;
        const double u2m  = su2  * inv_NxNy;
        const double vm   = sv   * inv_NxNy;
        const double v2m  = sv2  * inv_NxNy;
        const double wcm  = swc  * inv_NxNy;
        const double wc2m = swc2 * inv_NxNy;
        const double uwcm = suwc * inv_NxNy;
        const double pm   = sp   * inv_NxNy;

        // Welford incremental mean update
        U_m_  [kl0] += (um   - U_m_  [kl0]) * inv_n;
        U2_m_ [kl0] += (u2m  - U2_m_ [kl0]) * inv_n;
        V_m_  [kl0] += (vm   - V_m_  [kl0]) * inv_n;
        V2_m_ [kl0] += (v2m  - V2_m_ [kl0]) * inv_n;
        Wc_m_ [kl0] += (wcm  - Wc_m_ [kl0]) * inv_n;
        Wc2_m_[kl0] += (wc2m - Wc2_m_[kl0]) * inv_n;
        UWc_m_[kl0] += (uwcm - UWc_m_[kl0]) * inv_n;
        P_m_  [kl0] += (pm   - P_m_  [kl0]) * inv_n;
    }
}

// ============================================================
//  write — gather (8 Allreduce), rank0 writes Tecplot ASCII
// ============================================================
void Statistics::write(const std::string& path, int step, double Re_b,
                        bool reset_after)
{
    // All ranks gather (single MPI_Allreduce over 8 fields coalesced)
    std::vector<double> U_g, U2_g, V_g, V2_g, Wc_g, Wc2_g, UWc_g, P_g;
    {
        std::vector<const std::vector<double>*> locals = {
            &U_m_, &U2_m_, &V_m_, &V2_m_, &Wc_m_, &Wc2_m_, &UWc_m_, &P_m_
        };
        std::vector<std::vector<double>*> globals = {
            &U_g, &U2_g, &V_g, &V2_g, &Wc_g, &Wc2_g, &UWc_g, &P_g
        };
        gather_many(locals, globals, nz_global_, nz_local_, kstart_, topo_.cart());
    }

    const long n_saved = n_;
    if (reset_after) reset();

    if (topo_.rank() != 0 || n_saved == 0) return;

    // u_tau from time-averaged bottom-wall gradient: τ_w = ν·|<U(z≈0)>|/zc[0]
    const double nu    = 1.0 / Re_b;
    const double tau_w = nu * std::abs(U_g[0]) / zc_global_[0];
    const double u_tau = std::sqrt(std::max(tau_w, 0.0));
    const double inv_nu = Re_b;   // = 1/nu

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
        const double zp    = zc_global_[k] * u_tau * inv_nu;
        const double u_rms = std::sqrt(std::max(U2_g[k]  - U_g[k]  * U_g[k],  0.0));
        const double v_rms = std::sqrt(std::max(V2_g[k]  - V_g[k]  * V_g[k],  0.0));
        const double w_rms = std::sqrt(std::max(Wc2_g[k] - Wc_g[k] * Wc_g[k], 0.0));
        const double uw    = UWc_g[k] - U_g[k] * Wc_g[k];

        fprintf(fp, "%.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e\n",
                zc_global_[k], zp,
                U_g[k], Wc_g[k],
                u_rms, v_rms, w_rms,
                uw, P_g[k]);
    }
    fclose(fp);
}

} // namespace channel
