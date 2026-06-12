// channel/FieldOutput.cpp

#include "FieldOutput.hpp"

#include "Config.hpp"
#include "Grid.hpp"
#include "MpiTopology.hpp"
#include "Subdomain.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

namespace channel {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
FieldOutput::FieldOutput(const Config&      cfg,
                         const MpiTopology& topo,
                         const Subdomain&   sub,
                         const Grid&        grid)
    : cfg_(cfg), topo_(topo), sub_(sub), grid_(grid)
{
    const int nx    = sub_.nx();
    const int ny    = sub_.ny();
    const int nz    = sub_.nz();
    const int nx_g  = sub_.global_n(0);
    const int ny_g  = sub_.global_n(1);
    const int nz_g  = sub_.global_n(2);

    const auto& dx = grid_.dx(0);
    const auto& dy = grid_.dx(1);
    const auto& dz = grid_.dx(2);

    xc_.resize(nx_g + 2, 0.0);
    {
        double cum = 0.0;
        for (int i = 1; i <= nx_g; ++i) {
            xc_[i] = cum + dx[i] * 0.5;
            cum    += dx[i];
        }
    }

    yc_.resize(ny_g + 2, 0.0);
    {
        double cum = 0.0;
        for (int j = 1; j <= ny_g; ++j) {
            yc_[j] = cum + dy[j] * 0.5;
            cum    += dy[j];
        }
    }

    // Local zc_ with global z-offset applied (used for plane output)
    zc_.resize(nz + 2, 0.0);
    {
        const int k0 = sub_.ista(2) - 1;
        double cum = 0.0;
        for (int k = 1; k <= k0; ++k) cum += dz[k];
        for (int k = 1; k <= nz; ++k) {
            zc_[k] = cum + dz[k] * 0.5;
            cum    += dz[k];
        }
    }

    // XY plane at global k_mid
    {
        const int k_mid_g = (nz_g + 1) / 2;
        const int k_lo    = sub_.ista(2);
        const int k_hi    = k_lo + nz - 1;
        owns_xy_plane_ = (k_mid_g >= k_lo && k_mid_g <= k_hi);
        if (owns_xy_plane_)
            k_mid_local_ = k_mid_g - k_lo + 1;
    }

    j_mid_local_ = (ny + 1) / 2;
    i_mid_local_ = (nx_g + 1) / 2;

    {
        int nranks = 0;
        MPI_Comm_size(topo_.cart(), &nranks);
        if (nranks > 1) {
            char buf[16];
            std::snprintf(buf, sizeof(buf), "_r%03d", topo_.rank());
            rank_sfx_ = buf;
        }
    }
}

FieldOutput::~FieldOutput() = default;

// ---------------------------------------------------------------------------
// Helpers shared by write_planes
// ---------------------------------------------------------------------------
std::FILE* FieldOutput::open_plt_(const std::string& name) const
{
    std::string path = cfg_.dir_instantfield + name + rank_sfx_ + ".plt";
    return std::fopen(path.c_str(), "a");
}

void FieldOutput::zone_header_(std::FILE* fp, bool& first,
                               const char* vars,
                               int ni, int nj, int nk,
                               long step, double time) const
{
    if (first) {
        std::fprintf(fp, "VARIABLES=%s\n", vars);
        first = false;
    }
    std::fprintf(fp,
        "zone t=\"%ld\" i=%d j=%d k=%d STRANDID=%ld SOLUTIONTIME=%.6e\n",
        step, ni, nj, nk, step, time);
}

// ---------------------------------------------------------------------------
// write_planes — 2-D plane snapshots
// ---------------------------------------------------------------------------
void FieldOutput::write_planes(const Field<double>& U,
                               const Field<double>& V,
                               const Field<double>& W,
                               const Field<double>& P,
                               long step, double time)
{
    const int nx   = sub_.nx();
    const int ny   = sub_.ny();
    const int nz   = sub_.nz();
    const int nx_g = sub_.global_n(0);

    const char* vars = "\"X\" \"Y\" \"Z\" \"U\" \"V\" \"W\" \"P\"";

    // XY plane
    if (owns_xy_plane_) {
        std::FILE* fp = open_plt_("Output_instantfield_XY");
        if (fp) {
            const int k = k_mid_local_;
            zone_header_(fp, first_xy_, vars, nx, ny, 1, step, time);
            for (int j = 1; j <= ny; ++j)
                for (int i = 1; i <= nx; ++i)
                    std::fprintf(fp,
                        "%.6e %.6e %.6e %.6e %.6e %.6e %.6e\n",
                        xc_[i], yc_[j], zc_[k],
                        uc(U,i,j,k), vc(V,i,j,k), wc(W,i,j,k), P(i,j,k));
            std::fclose(fp);
        }
    }

    // XZ plane
    {
        std::FILE* fp = open_plt_("Output_instantfield_XZ");
        if (fp) {
            const int j = j_mid_local_;
            zone_header_(fp, first_xz_, vars, nx, 1, nz, step, time);
            for (int k = 1; k <= nz; ++k)
                for (int i = 1; i <= nx; ++i)
                    std::fprintf(fp,
                        "%.6e %.6e %.6e %.6e %.6e %.6e %.6e\n",
                        xc_[i], yc_[j], zc_[k],
                        uc(U,i,j,k), vc(V,i,j,k), wc(W,i,j,k), P(i,j,k));
            std::fclose(fp);
        }
    }

    // YZ plane
    {
        const int i = i_mid_local_;
        if (i >= 1 && i <= nx_g) {
            std::FILE* fp = open_plt_("Output_instantfield_YZ");
            if (fp) {
                zone_header_(fp, first_yz_, vars, 1, ny, nz, step, time);
                for (int k = 1; k <= nz; ++k)
                    for (int j = 1; j <= ny; ++j)
                        std::fprintf(fp,
                            "%.6e %.6e %.6e %.6e %.6e %.6e %.6e\n",
                            xc_[i], yc_[j], zc_[k],
                            uc(U,i,j,k), vc(V,i,j,k), wc(W,i,j,k), P(i,j,k));
                std::fclose(fp);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// compute_lambda2_ — middle eigenvalue of S²+Ω² via Cardano (analytical)
// ---------------------------------------------------------------------------
double FieldOutput::compute_lambda2_(double M11, double M22, double M33,
                                     double M12, double M13, double M23)
{
    double p1 = M12*M12 + M13*M13 + M23*M23;
    if (p1 < 1.0e-30) {
        double e[3] = {M11, M22, M33};
        std::sort(e, e + 3);
        return e[1];
    }
    double q  = (M11 + M22 + M33) / 3.0;
    double b11 = M11 - q, b22 = M22 - q, b33 = M33 - q;
    double p2 = b11*b11 + b22*b22 + b33*b33 + 2.0*p1;
    double p  = std::sqrt(p2 / 6.0);
    if (p < 1.0e-30) return q;
    double B11 = b11/p, B22 = b22/p, B33 = b33/p;
    double B12 = M12/p, B13 = M13/p, B23 = M23/p;
    double r = 0.5 * (B11*(B22*B33 - B23*B23)
                    - B12*(B12*B33 - B23*B13)
                    + B13*(B12*B23 - B22*B13));
    r = std::max(-1.0, std::min(1.0, r));
    double phi  = std::acos(r) / 3.0;
    const double two_pi_3 = 2.0 * M_PI / 3.0;
    double eig1 = q + 2.0*p*std::cos(phi);
    double eig3 = q + 2.0*p*std::cos(phi + two_pi_3);
    return 3.0*q - eig1 - eig3;   // middle eigenvalue
}

// ---------------------------------------------------------------------------
// write_field_3d — full 3-D Tecplot ASCII, rank-ordered sequential write
//
// Variables: X Y Z U V W P Q Lambda2
// U,V,W are interpolated to cell centres.
// Velocity gradients: diagonal via face-to-face (exact), off-diagonal via
// central differences of cell-centre velocities (clamped at boundaries).
// ---------------------------------------------------------------------------
void FieldOutput::write_field_3d(const Field<double>& U,
                                  const Field<double>& V,
                                  const Field<double>& W,
                                  const Field<double>& P,
                                  int step)
{
    const int nx   = sub_.nx();
    const int ny   = sub_.ny();
    const int nz   = sub_.nz();
    const int nx_g = sub_.global_n(0);
    const int ny_g = sub_.global_n(1);
    const int nz_g = sub_.global_n(2);

    int nranks = 0, myrank = 0;
    MPI_Comm_size(topo_.cart(), &nranks);
    MPI_Comm_rank(topo_.cart(), &myrank);

    // Build output path
    char fname[256];
    std::snprintf(fname, sizeof(fname), "%sOutput_field_%08d.plt",
                  cfg_.dir_instantfield.c_str(), step);

    // Grid arrays (local, includes ghost cells at 0 and n+1)
    const auto& gx  = grid_.x(0);   // cell-centre x, 1-indexed local
    const auto& gy  = grid_.x(1);   // cell-centre y
    const auto& gz  = grid_.x(2);   // cell-centre z (with ghost via Grid mirrors)
    const auto& gdx = grid_.dx(0);  // cell width x
    const auto& gdy = grid_.dx(1);  // cell width y
    const auto& gdz = grid_.dx(2);  // cell width z

    // Rank 0 creates the file and writes the header
    if (myrank == 0) {
        FILE* fp = fopen(fname, "w");
        if (!fp) {
            fprintf(stderr, "[FieldOutput::write_field_3d] Cannot open %s\n", fname);
            MPI_Abort(topo_.cart(), 1);
        }
        fprintf(fp, "TITLE = \"Channel Flow Field (step=%d)\"\n", step);
        fprintf(fp,
            "VARIABLES = \"X\" \"Y\" \"Z\" \"U\" \"V\" \"W\" \"P\" \"Q\" \"Lambda2\"\n");
        fprintf(fp, "ZONE T=\"Field\", I=%d, J=%d, K=%d, DATAPACKING=POINT\n",
                nx_g, ny_g, nz_g);
        fclose(fp);
    }
    MPI_Barrier(topo_.cart());

    // Each rank appends its z-slab in rank order
    for (int r = 0; r < nranks; ++r) {
        if (myrank == r) {
            FILE* fp = fopen(fname, "a");
            if (!fp) {
                fprintf(stderr,
                    "[FieldOutput::write_field_3d] Cannot open %s (rank %d)\n",
                    fname, myrank);
                MPI_Abort(topo_.cart(), 1);
            }

            for (int k = 1; k <= nz; ++k) {
                // inv_dz_face: for exact dw/dz from z-faces
                const double inv_dz_face = 1.0 / gdz[k];
                // For central diff in z (du/dz, dv/dz): use cell-centre positions
                // gz[k-1] and gz[k+1] are valid (Grid sets ghost values)
                const double inv_2dz = 0.5 / (gz[k+1] - gz[k-1]);

                for (int j = 1; j <= ny; ++j) {
                    // Clamp j for off-diagonal gradients (avoids corner ghosts)
                    const int jm = std::max(j - 1, 1);
                    const int jp = std::min(j + 1, ny);
                    const double inv_2dy = 0.5 / (gy[jp] - gy[jm]);

                    for (int i = 1; i <= nx; ++i) {
                        // Clamp i (avoids corner ghosts; x is periodic but
                        // second ghost not guaranteed after halo)
                        const int im = std::max(i - 1, 1);
                        const int ip = std::min(i + 1, nx);
                        const double inv_2dx = 0.5 / (gx[ip] - gx[im]);

                        // Cell-centre velocities at (i,j,k) and neighbours
                        // uc(i,j,k) = 0.5*(U(i,j,k)+U(i+1,j,k)) — needs no corners
                        const double ui   = uc(U, i,  j,  k );
                        const double ui_p = uc(U, ip, j,  k );
                        const double ui_m = uc(U, im, j,  k );
                        const double ui_jp= uc(U, i,  jp, k );
                        const double ui_jm= uc(U, i,  jm, k );
                        const double ui_kp= uc(U, i,  j,  k+1);
                        const double ui_km= uc(U, i,  j,  k-1);

                        const double vi   = vc(V, i,  j,  k );
                        const double vi_p = vc(V, ip, j,  k );
                        const double vi_m = vc(V, im, j,  k );
                        const double vi_jp= vc(V, i,  jp, k );
                        const double vi_jm= vc(V, i,  jm, k );
                        const double vi_kp= vc(V, i,  j,  k+1);
                        const double vi_km= vc(V, i,  j,  k-1);

                        const double wi   = wc(W, i,  j,  k );
                        const double wi_p = wc(W, ip, j,  k );
                        const double wi_m = wc(W, im, j,  k );
                        const double wi_jp= wc(W, i,  jp, k );
                        const double wi_jm= wc(W, i,  jm, k );

                        // Velocity gradient tensor (all 9 components)
                        const double dudx = (U(i+1,j,k) - U(i,j,k)) * (1.0/gdx[i]);
                        const double dvdy = (V(i,j+1,k) - V(i,j,k)) * (1.0/gdy[j]);
                        const double dwdz = (W(i,j,k+1) - W(i,j,k)) * inv_dz_face;

                        const double dudy = (ui_jp - ui_jm) * inv_2dy;
                        const double dudz = (ui_kp - ui_km) * inv_2dz;
                        const double dvdx = (vi_p  - vi_m ) * inv_2dx;
                        const double dvdz = (vi_kp - vi_km) * inv_2dz;
                        const double dwdx = (wi_p  - wi_m ) * inv_2dx;
                        const double dwdy = (wi_jp - wi_jm) * inv_2dy;

                        // Strain rate S_ij and vorticity Ω_ij
                        const double S11 = dudx, S22 = dvdy, S33 = dwdz;
                        const double S12 = 0.5*(dudy + dvdx);
                        const double S13 = 0.5*(dudz + dwdx);
                        const double S23 = 0.5*(dvdz + dwdy);
                        const double W12 = 0.5*(dudy - dvdx);
                        const double W13 = 0.5*(dudz - dwdx);
                        const double W23 = 0.5*(dvdz - dwdy);

                        // Q = 0.5*(||Ω||² - ||S||²)
                        const double normS2 = S11*S11 + S22*S22 + S33*S33
                                            + 2.0*(S12*S12 + S13*S13 + S23*S23);
                        const double normW2 = 2.0*(W12*W12 + W13*W13 + W23*W23);
                        const double Q = 0.5*(normW2 - normS2);

                        // M = S² + Ω² (symmetric 3×3)
                        const double M11 = S11*S11 + S12*S12 + S13*S13
                                         - W12*W12 - W13*W13;
                        const double M22 = S12*S12 + S22*S22 + S23*S23
                                         - W12*W12 - W23*W23;
                        const double M33 = S13*S13 + S23*S23 + S33*S33
                                         - W13*W13 - W23*W23;
                        const double M12 = S11*S12 + S12*S22 + S13*S23 - W13*W23;
                        const double M13 = S11*S13 + S12*S23 + S13*S33 + W12*W23;
                        const double M23 = S12*S13 + S22*S23 + S23*S33 - W12*W13;

                        const double lam2 = compute_lambda2_(M11,M22,M33,M12,M13,M23);

                        // Use global xc_[i], yc_[j] for output coordinates
                        fprintf(fp,
                            "%.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e\n",
                            xc_[sub_.ista(0) + i - 1],
                            yc_[sub_.ista(1) + j - 1],
                            zc_[k],
                            ui, vi, wi, P(i,j,k),
                            Q, lam2);
                    }
                }
            }
            fclose(fp);
        }
        MPI_Barrier(topo_.cart());
    }
}

} // namespace channel
