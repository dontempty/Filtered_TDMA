#include "solve_theta.hpp"
#include "tdma_backend.hpp"
#include "index.hpp"
#include "stencil_coeffs.hpp"
#include "debug.hpp"
#include "timing_csv.hpp"

#include <cstdlib>
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

    const bool do_timing = (params_.option != "order");
    const int n_warmup   = params_.Nt_warmup;
    const int n_timing   = max_iter - n_warmup;
    const std::vector<std::string> event_names =
        {"rhs", "solve_z", "solve_y", "solve_x", "comm"};
    const int n_events = (int)event_names.size();
    if (do_timing)
        timing_csv::timing_init(n_events, n_timing, MPI_COMM_WORLD);
    std::vector<double> local_times(n_events, 0.0);

    // Uniform stencil (ghost-cell approach): same coefficients everywhere.
    const auto sx0 = compute_stencil(dt, sub_.dmx_sub[1]);
    const auto sy0 = compute_stencil(dt, sub_.dmy_sub[1]);
    const auto sz0 = compute_stencil(dt, sub_.dmz_sub[1]);

    for (int t_step = 0; t_step < max_iter; ++t_step) {

        solver_x.set_eps_constant(dt);
        solver_y.set_eps_constant(dt);
        solver_z.set_eps_constant(dt);

        // Update physical-boundary ghost cells: u_ghost = 2*u_BC - u_adjacent
        sub_.boundary(theta);

        double rhs_t0, rhs_t1;
        double solve_z_t0, solve_z_t1;
        double solve_y_t0, solve_y_t1;
        double solve_x_t0, solve_x_t1;
        double comm_t0, comm_t1;

        // ============================================================
        // RHS computation (ghost cells already set in theta[])
        // ============================================================
        rhs_t0 = MPI_Wtime();
        for (k = 1; k < nz1-1; ++k) {
            for (j = 1; j < ny1-1; ++j) {
                for (i = 1; i < nx1-1; ++i) {
                    ijk    = idx_ijk(i,   j,   k,   nx1, ny1);
                    int ip = idx_ijk(i+1, j,   k,   nx1, ny1);
                    int im = idx_ijk(i-1, j,   k,   nx1, ny1);
                    int jp = idx_ijk(i,   j+1, k,   nx1, ny1);
                    int jm = idx_ijk(i,   j-1, k,   nx1, ny1);
                    int kp = idx_ijk(i,   j,   k+1, nx1, ny1);
                    int km = idx_ijk(i,   j,   k-1, nx1, ny1);

                    rhs[ijk] = (sx0.c*theta[ip] + sx0.a*theta[im])
                             + (sy0.c*theta[jp] + sy0.a*theta[jm])
                             + (sz0.c*theta[kp] + sz0.a*theta[km])
                             + (1.0 + sx0.b + sy0.b + sz0.b) * theta[ijk]
                             + dt * 3.0 * Pi*Pi * cos(Pi*sub_.x_sub[i])
                                                * cos(Pi*sub_.y_sub[j])
                                                * cos(Pi*sub_.z_sub[k]);
                }
            }
        }
        rhs_t1 = MPI_Wtime();

        // ============================================================
        // Z solve — build TDMA then add implicit ghost correction
        // ============================================================
        MPI_Barrier(MPI_COMM_WORLD);
        solve_z_t0 = MPI_Wtime();
        for (j = 1; j < ny1-1; ++j) {
            for (k = 1; k < nz1-1; ++k) {
                for (i = 1; i < nx1-1; ++i) {
                    int ik = idx_ik(i-1, k-1, nx1-2);
                    Azz[ik] = -sz0.a;
                    Bzz[ik] = 1.0 - sz0.b;
                    Czz[ik] = -sz0.c;
                    Dzz[ik] = rhs[idx_ijk(i, j, k, nx1, ny1)];
                }
            }
            // Fully-implicit Dirichlet wall: fold the ghost's unknown part into
            // the diagonal (1+2base -> 1+3base) instead of lagging it at theta^n;
            // move only the known 2*uBC = (ghost + boundary cell) to the RHS.
            if (sub_.theta_z_left_index[1]) {
                for (i = 1; i < nx1-1; ++i) {
                    int row = idx_ik(i-1, 0, nx1-2);
                    Dzz[row] += sz0.a * (theta[idx_ijk(i, j, 0, nx1, ny1)]
                                       + theta[idx_ijk(i, j, 1, nx1, ny1)]);
                    Bzz[row] += sz0.a;
                }
            }
            if (sub_.theta_z_right_index[nz1-2]) {
                for (i = 1; i < nx1-1; ++i) {
                    int row = idx_ik(i-1, nz1-3, nx1-2);
                    Dzz[row] += sz0.c * (theta[idx_ijk(i, j, nz1-1, nx1, ny1)]
                                       + theta[idx_ijk(i, j, nz1-2, nx1, ny1)]);
                    Bzz[row] += sz0.c;
                }
            }
            solver_z.set_rho(Azz.data(), Bzz.data(), Czz.data());
            if (params_.periodic[2])
                solver_z.solve_cyclic(Azz.data(), Bzz.data(), Czz.data(), Dzz.data());
            else
                solver_z.solve(Azz.data(), Bzz.data(), Czz.data(), Dzz.data());
            for (k = 1; k < nz1-1; ++k)
                for (i = 1; i < nx1-1; ++i)
                    rhs[idx_ijk(i, j, k, nx1, ny1)] = Dzz[idx_ik(i-1, k-1, nx1-2)];
        }
        solve_z_t1 = MPI_Wtime();

        // ============================================================
        // Y solve
        // ============================================================
        MPI_Barrier(MPI_COMM_WORLD);
        solve_y_t0 = MPI_Wtime();
        for (k = 1; k < nz1-1; ++k) {
            for (j = 1; j < ny1-1; ++j) {
                for (i = 1; i < nx1-1; ++i) {
                    int ij = idx_ij(i-1, j-1, nx1-2);
                    Ayy[ij] = -sy0.a;
                    Byy[ij] = 1.0 - sy0.b;
                    Cyy[ij] = -sy0.c;
                    Dyy[ij] = rhs[idx_ijk(i, j, k, nx1, ny1)];
                }
            }
            if (sub_.theta_y_left_index[1]) {
                for (i = 1; i < nx1-1; ++i) {
                    int row = idx_ij(i-1, 0, nx1-2);
                    Dyy[row] += sy0.a * (theta[idx_ijk(i, 0, k, nx1, ny1)]
                                       + theta[idx_ijk(i, 1, k, nx1, ny1)]);
                    Byy[row] += sy0.a;
                }
            }
            if (sub_.theta_y_right_index[ny1-2]) {
                for (i = 1; i < nx1-1; ++i) {
                    int row = idx_ij(i-1, ny1-3, nx1-2);
                    Dyy[row] += sy0.c * (theta[idx_ijk(i, ny1-1, k, nx1, ny1)]
                                       + theta[idx_ijk(i, ny1-2, k, nx1, ny1)]);
                    Byy[row] += sy0.c;
                }
            }
            solver_y.set_rho(Ayy.data(), Byy.data(), Cyy.data());
            if (params_.periodic[1])
                solver_y.solve_cyclic(Ayy.data(), Byy.data(), Cyy.data(), Dyy.data());
            else
                solver_y.solve(Ayy.data(), Byy.data(), Cyy.data(), Dyy.data());
            for (j = 1; j < ny1-1; ++j)
                for (i = 1; i < nx1-1; ++i)
                    rhs[idx_ijk(i, j, k, nx1, ny1)] = Dyy[idx_ij(i-1, j-1, nx1-2)];
        }
        solve_y_t1 = MPI_Wtime();

        // ============================================================
        // X solve
        // ============================================================
        MPI_Barrier(MPI_COMM_WORLD);
        solve_x_t0 = MPI_Wtime();
        for (k = 1; k < nz1-1; ++k) {
            for (i = 1; i < nx1-1; ++i) {
                for (j = 1; j < ny1-1; ++j) {
                    int ji = idx_ji(j-1, i-1, ny1-2);
                    Axx[ji] = -sx0.a;
                    Bxx[ji] = 1.0 - sx0.b;
                    Cxx[ji] = -sx0.c;
                    Dxx[ji] = rhs[idx_ijk(i, j, k, nx1, ny1)];
                }
            }
            if (sub_.theta_x_left_index[1]) {
                for (j = 1; j < ny1-1; ++j) {
                    int row = idx_ji(j-1, 0, ny1-2);
                    Dxx[row] += sx0.a * (theta[idx_ijk(0, j, k, nx1, ny1)]
                                       + theta[idx_ijk(1, j, k, nx1, ny1)]);
                    Bxx[row] += sx0.a;
                }
            }
            if (sub_.theta_x_right_index[nx1-2]) {
                for (j = 1; j < ny1-1; ++j) {
                    int row = idx_ji(j-1, nx1-3, ny1-2);
                    Dxx[row] += sx0.c * (theta[idx_ijk(nx1-1, j, k, nx1, ny1)]
                                       + theta[idx_ijk(nx1-2, j, k, nx1, ny1)]);
                    Bxx[row] += sx0.c;
                }
            }
            solver_x.set_rho(Axx.data(), Bxx.data(), Cxx.data());
            if (params_.periodic[0])
                solver_x.solve_cyclic(Axx.data(), Bxx.data(), Cxx.data(), Dxx.data());
            else
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
        // Record timing (scaling tests only, after warmup)
        // ============================================================
        if (do_timing) {
            local_times[0] = rhs_t1    - rhs_t0;
            local_times[1] = solve_z_t1 - solve_z_t0;
            local_times[2] = solve_y_t1 - solve_y_t0;
            local_times[3] = solve_x_t1 - solve_x_t0;
            local_times[4] = comm_t1   - comm_t0;
            if (t_step >= n_warmup)
                timing_csv::timing_record(t_step - n_warmup, local_times, MPI_COMM_WORLD);
        }

    } // end time loop

    // ------------------------------------------------------------------
    //  Write CSV (scaling tests only).
    //  Path:  results/timing_<nx>_<npx><npy><npz>_<backend>.csv
    // ------------------------------------------------------------------
    if (do_timing) {
        char meta[256];
        std::snprintf(meta, sizeof(meta),
                      "grid=%dx%dx%d, np=%d (%d,%d,%d), dt=%10.3E, Nt=%d, solver_kind=%s",
                      params_.nx, params_.ny, params_.nz,
                      params_.np_dim[0] * params_.np_dim[1] * params_.np_dim[2],
                      params_.np_dim[0], params_.np_dim[1], params_.np_dim[2],
                      dt, max_iter, params_.tdma_backend.c_str());

        char fn[256];
        const char* env_path = std::getenv("TIMING_CSV");
        if (env_path && env_path[0] != '\0') {
            std::snprintf(fn, sizeof(fn), "%s", env_path);
        } else {
            std::snprintf(fn, sizeof(fn), "results/timing_%d_%d%d%d_%s.csv",
                          params_.nx,
                          params_.np_dim[0], params_.np_dim[1], params_.np_dim[2],
                          params_.tdma_backend.c_str());
        }

        timing_csv::timing_save_csv(fn, event_names, meta, MPI_COMM_WORLD);
    }
    timing_csv::timing_cleanup();
}
