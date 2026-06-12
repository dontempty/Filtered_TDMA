#include "solve_theta.hpp"
#include "tdma_backend.hpp"
#include "index.hpp"
#include "stencil_coeffs.hpp"
#include "debug.hpp"

#include <iostream>
#include <string>
#include <cmath>

SolveTheta::SolveTheta(const GlobalParams& params,
                       const MPITopology& topo,
                       MPISubdomain& sub)
    : params_(params), topo_(topo), sub_(sub) {}

void SolveTheta::run(std::vector<double>& theta) {

    int nz1 = sub_.nz_sub + 1;
    int ny1 = sub_.ny_sub + 1;
    int nx1 = sub_.nx_sub + 1;

    int i, j, k, ijk;
    int idx, idx_ip, idx_im;

    double dt = params_.dt;
    int max_iter = params_.Nt;
    double eps_c = params_.eps_constant;

    // --- Create solvers for each direction ---
    auto cx = topo_.commX();
    auto cy = topo_.commY();
    auto cz = topo_.commZ();

    auto backend_kind = TdmaBackend::parse(params_.tdma_backend);
    TdmaBackend solver_x(backend_kind, ny1-2, nx1-2,
                         cx.myrank, cx.nprocs, cx.comm,
                         cx.west_rank, cx.east_rank, eps_c);
    TdmaBackend solver_y(backend_kind, nx1-2, ny1-2,
                         cy.myrank, cy.nprocs, cy.comm,
                         cy.west_rank, cy.east_rank, eps_c);
    TdmaBackend solver_z(backend_kind, nx1-2, nz1-2,
                         cz.myrank, cz.nprocs, cz.comm,
                         cz.west_rank, cz.east_rank, eps_c);

    std::vector<double> Axx((nx1-2)*(ny1-2)), Bxx((nx1-2)*(ny1-2));
    std::vector<double> Cxx((nx1-2)*(ny1-2)), Dxx((nx1-2)*(ny1-2));
    std::vector<double> Ayy((ny1-2)*(nx1-2)), Byy((ny1-2)*(nx1-2));
    std::vector<double> Cyy((ny1-2)*(nx1-2)), Dyy((ny1-2)*(nx1-2));
    std::vector<double> Azz((nz1-2)*(nx1-2)), Bzz((nz1-2)*(nx1-2));
    std::vector<double> Czz((nz1-2)*(nx1-2)), Dzz((nz1-2)*(nx1-2));

    std::vector<double> rhs(nx1 * ny1 * nz1, 0.0);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if (my_rank == 0) {
        std::cout << "[Tmax] = " << dt * max_iter
                  << " | [max_iter] = " << max_iter << "\n";
        std::cout << "[nx_sub] = " << nx1-2 << " | [ny_sub] = " << ny1-2
                  << " | [nz_sub] = " << nz1-2 << "\n";
        double beta = dt / 2.0 / (sub_.dmz_sub[1] * sub_.dmz_sub[1]);
        std::cout << "[rho] = " << beta / (1.0 + 2.0*beta) << "\n";
        std::cout << "[eps const] = " << eps_c << "\n";
        std::cout << "[backend] = " << params_.tdma_backend << "\n";
    }

    for (int t_step = 0; t_step < max_iter; ++t_step) {

        // Refresh FilteredTDMA truncation threshold each step (channel pattern:
        // eps_ = dt / (N*N) inside the library — see filtered_tdma.hpp).
        // No-op for PaScaL backend.
        solver_x.set_eps_constant(dt);
        solver_y.set_eps_constant(dt);
        solver_z.set_eps_constant(dt);

        double rhs_t0, rhs_t1;
        double solve_z_t0, solve_z_t1;
        double solve_y_t0, solve_y_t1;
        double solve_x_t0, solve_x_t1;
        double comm_t0, comm_t1;

        // ============================================================
        // RHS computation
        // ============================================================
        rhs_t0 = MPI_Wtime();
        for (k = 1; k < nz1-1; ++k) {
            auto sz = compute_stencil(dt, sub_.dmz_sub[k],
                                      sub_.theta_z_left_index[k],
                                      sub_.theta_z_right_index[k]);
            for (j = 1; j < ny1-1; ++j) {
                auto sy = compute_stencil(dt, sub_.dmy_sub[j],
                                          sub_.theta_y_left_index[j],
                                          sub_.theta_y_right_index[j]);
                for (i = 1; i < nx1-1; ++i) {
                    auto sx = compute_stencil(dt, sub_.dmx_sub[i],
                                              sub_.theta_x_left_index[i],
                                              sub_.theta_x_right_index[i]);

                    ijk          = idx_ijk(i,   j,   k,   nx1, ny1);
                    int ip       = idx_ijk(i+1, j,   k,   nx1, ny1);
                    int im       = idx_ijk(i-1, j,   k,   nx1, ny1);
                    int jp       = idx_ijk(i,   j+1, k,   nx1, ny1);
                    int jm       = idx_ijk(i,   j-1, k,   nx1, ny1);
                    int kp       = idx_ijk(i,   j,   k+1, nx1, ny1);
                    int km       = idx_ijk(i,   j,   k-1, nx1, ny1);

                    rhs[ijk] = (sx.c*theta[ip] + sx.a*theta[im])
                             + (sy.c*theta[jp] + sy.a*theta[jm])
                             + (sz.c*theta[kp] + sz.a*theta[km])
                             + (1.0 + sx.b + sy.b + sz.b) * theta[ijk]
                             + dt * 3.0 * Pi*Pi * cos(Pi*sub_.x_sub[i])
                                                * cos(Pi*sub_.y_sub[j])
                                                * cos(Pi*sub_.z_sub[k]);
                }
            }
        }
        rhs_t1 = MPI_Wtime();

        // ============================================================
        // Z-boundary correction
        // ============================================================
        for (j = 1; j < ny1-1; ++j) {
            auto sy = compute_stencil(dt, sub_.dmy_sub[j],
                                      sub_.theta_y_left_index[j],
                                      sub_.theta_y_right_index[j]);
            for (i = 1; i < nx1-1; ++i) {
                auto sx = compute_stencil(dt, sub_.dmx_sub[i],
                                          sub_.theta_x_left_index[i],
                                          sub_.theta_x_right_index[i]);

                // k=0 boundary
                double dz0 = sub_.dmz_sub[0];
                double coef_za = (dt / 2.0 / (dz0*dz0)) * (1.0 + 5.0/3.0);

                for (int dj = -1; dj <= 1; ++dj) {
                    double cy_val = (dj == -1) ? -sy.a : (dj == 0) ? (1.0-sy.b) : -sy.c;
                    idx    = idx_ij(i,   j+dj, nx1);
                    idx_ip = idx_ij(i+1, j+dj, nx1);
                    idx_im = idx_ij(i-1, j+dj, nx1);
                    rhs[idx_ijk(i, j, 1, nx1, ny1)] +=
                        coef_za * cy_val *
                        (-sx.a * sub_.theta_z_left_sub[idx_im]
                       + (1.0 - sx.b) * sub_.theta_z_left_sub[idx]
                       - sx.c * sub_.theta_z_left_sub[idx_ip])
                        * sub_.theta_z_left_index[1];
                }

                // k=nz1-1 boundary
                double dzN = sub_.dmz_sub[nz1-1];
                double coef_zc = (dt / 2.0 / (dzN*dzN)) * (1.0 + 5.0/3.0);

                for (int dj = -1; dj <= 1; ++dj) {
                    double cy_val = (dj == -1) ? -sy.a : (dj == 0) ? (1.0-sy.b) : -sy.c;
                    idx    = idx_ij(i,   j+dj, nx1);
                    idx_ip = idx_ij(i+1, j+dj, nx1);
                    idx_im = idx_ij(i-1, j+dj, nx1);
                    rhs[idx_ijk(i, j, nz1-2, nx1, ny1)] +=
                        coef_zc * cy_val *
                        (-sx.a * sub_.theta_z_right_sub[idx_im]
                       + (1.0 - sx.b) * sub_.theta_z_right_sub[idx]
                       - sx.c * sub_.theta_z_right_sub[idx_ip])
                        * sub_.theta_z_right_index[nz1-2];
                }
            }
        }

        // ============================================================
        // Z solve
        // ============================================================
        MPI_Barrier(MPI_COMM_WORLD);
        solve_z_t0 = MPI_Wtime();
        for (j = 1; j < ny1-1; ++j) {
            for (k = 1; k < nz1-1; ++k) {
                auto sz = compute_stencil(dt, sub_.dmz_sub[k],
                                          sub_.theta_z_left_index[k],
                                          sub_.theta_z_right_index[k]);
                for (i = 1; i < nx1-1; ++i) {
                    ijk = idx_ijk(i, j, k, nx1, ny1);
                    int ik = idx_ik(i-1, k-1, nx1-2);
                    Azz[ik] = -sz.a;
                    Bzz[ik] = 1.0 - sz.b;
                    Czz[ik] = -sz.c;
                    Dzz[ik] = rhs[ijk];
                }
            }
            solver_z.set_rho(Azz.data(), Bzz.data(), Czz.data());
            solver_z.solve(Azz.data(), Bzz.data(), Czz.data(), Dzz.data());
            for (k = 1; k < nz1-1; ++k)
                for (i = 1; i < nx1-1; ++i)
                    rhs[idx_ijk(i, j, k, nx1, ny1)] = Dzz[idx_ik(i-1, k-1, nx1-2)];
        }
        solve_z_t1 = MPI_Wtime();

        // ============================================================
        // Y-boundary correction
        // ============================================================
        for (k = 1; k < nz1-1; ++k) {
            for (i = 1; i < nx1-1; ++i) {
                auto sx = compute_stencil(dt, sub_.dmx_sub[i],
                                          sub_.theta_x_left_index[i],
                                          sub_.theta_x_right_index[i]);
                // j=0
                double dy0 = sub_.dmy_sub[0];
                double coef_ya = (dt / 2.0 / (dy0*dy0)) * (1.0 + 5.0/3.0);
                idx    = idx_ik(i,   k, nx1);
                idx_ip = idx_ik(i+1, k, nx1);
                idx_im = idx_ik(i-1, k, nx1);
                rhs[idx_ijk(i, 1, k, nx1, ny1)] +=
                    coef_ya * sub_.theta_y_left_index[1] *
                    (-sx.a * sub_.theta_y_left_sub[idx_im]
                   + (1.0-sx.b) * sub_.theta_y_left_sub[idx]
                   - sx.c * sub_.theta_y_left_sub[idx_ip]);

                // j=ny1-1
                double dyN = sub_.dmy_sub[ny1-1];
                double coef_yc = (dt / 2.0 / (dyN*dyN)) * (1.0 + 5.0/3.0);
                rhs[idx_ijk(i, ny1-2, k, nx1, ny1)] +=
                    coef_yc * sub_.theta_y_right_index[ny1-2] *
                    (-sx.a * sub_.theta_y_right_sub[idx_im]
                   + (1.0-sx.b) * sub_.theta_y_right_sub[idx]
                   - sx.c * sub_.theta_y_right_sub[idx_ip]);
            }
        }

        // ============================================================
        // Y solve
        // ============================================================
        MPI_Barrier(MPI_COMM_WORLD);
        solve_y_t0 = MPI_Wtime();
        for (k = 1; k < nz1-1; ++k) {
            for (j = 1; j < ny1-1; ++j) {
                auto sy = compute_stencil(dt, sub_.dmy_sub[j],
                                          sub_.theta_y_left_index[j],
                                          sub_.theta_y_right_index[j]);
                for (i = 1; i < nx1-1; ++i) {
                    ijk = idx_ijk(i, j, k, nx1, ny1);
                    int ij = idx_ij(i-1, j-1, nx1-2);
                    Ayy[ij] = -sy.a;
                    Byy[ij] = 1.0 - sy.b;
                    Cyy[ij] = -sy.c;
                    Dyy[ij] = rhs[ijk];
                }
            }
            solver_y.set_rho(Ayy.data(), Byy.data(), Cyy.data());
            solver_y.solve(Ayy.data(), Byy.data(), Cyy.data(), Dyy.data());
            for (j = 1; j < ny1-1; ++j)
                for (i = 1; i < nx1-1; ++i)
                    rhs[idx_ijk(i, j, k, nx1, ny1)] = Dyy[idx_ij(i-1, j-1, nx1-2)];
        }
        solve_y_t1 = MPI_Wtime();

        // ============================================================
        // X-boundary correction
        // ============================================================
        for (k = 1; k < nz1-1; ++k) {
            for (j = 1; j < ny1-1; ++j) {
                // i=0
                double dx0 = sub_.dmx_sub[0];
                double coef_xa = (dt / 2.0 / (dx0*dx0)) * (1.0 + 5.0/3.0);
                idx = idx_jk(j, k, ny1);
                rhs[idx_ijk(1, j, k, nx1, ny1)] +=
                    coef_xa * sub_.theta_x_left_index[1] * sub_.theta_x_left_sub[idx];

                // i=nx1-1
                double dxN = sub_.dmx_sub[nx1-1];
                double coef_xc = (dt / 2.0 / (dxN*dxN)) * (1.0 + 5.0/3.0);
                rhs[idx_ijk(nx1-2, j, k, nx1, ny1)] +=
                    coef_xc * sub_.theta_x_right_index[nx1-2] * sub_.theta_x_right_sub[idx];
            }
        }

        // ============================================================
        // X solve
        // ============================================================
        MPI_Barrier(MPI_COMM_WORLD);
        solve_x_t0 = MPI_Wtime();
        for (k = 1; k < nz1-1; ++k) {
            for (i = 1; i < nx1-1; ++i) {
                auto sx = compute_stencil(dt, sub_.dmx_sub[i],
                                          sub_.theta_x_left_index[i],
                                          sub_.theta_x_right_index[i]);
                for (j = 1; j < ny1-1; ++j) {
                    ijk = idx_ijk(i, j, k, nx1, ny1);
                    int ji = idx_ji(j-1, i-1, ny1-2);
                    Axx[ji] = -sx.a;
                    Bxx[ji] = 1.0 - sx.b;
                    Cxx[ji] = -sx.c;
                    Dxx[ji] = rhs[ijk];
                }
            }
            solver_x.set_rho(Axx.data(), Bxx.data(), Cxx.data());
            solver_x.solve(Axx.data(), Bxx.data(), Cxx.data(), Dxx.data());
            for (i = 1; i < nx1-1; ++i)
                for (j = 1; j < ny1-1; ++j)
                    theta[idx_ijk(i, j, k, nx1, ny1)] = Dxx[idx_ji(j-1, i-1, ny1-2)];
        }
        solve_x_t1 = MPI_Wtime();

        // ============================================================
        // Ghost-cell update
        // ============================================================
        MPI_Barrier(MPI_COMM_WORLD);
        comm_t0 = MPI_Wtime();
        sub_.ghostcellUpdate(theta, cx, cy, cz, params_);
        comm_t1 = MPI_Wtime();

        // ============================================================
        // Save timing data (strong scaling mode)
        // ============================================================
        if (params_.option == "strong") {
            int nptot = params_.np_dim[0] * params_.np_dim[1] * params_.np_dim[2];
            save_timing_data(
                "results/t_" + std::to_string(nptot) + "_" + std::to_string(t_step) + ".txt",
                MPI_COMM_WORLD,
                {"rhs", "solve_z", "solve_y", "solve_x", "comm"},
                {rhs_t1 - rhs_t0,
                 solve_z_t1 - solve_z_t0,
                 solve_y_t1 - solve_y_t0,
                 solve_x_t1 - solve_x_t0,
                 comm_t1 - comm_t0}
            );
        }

    } // end time loop
}
