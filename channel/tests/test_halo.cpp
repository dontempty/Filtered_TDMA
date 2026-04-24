// tests/test_halo.cpp
//
// Round-trip test for HaloExchanger:
//   1) Each rank fills the interior of a Field with a known global
//      pattern f(I, J, K) where (I, J, K) is the global cell index.
//   2) After exchange(), every interior cell still equals its local pattern,
//      and every halo cell equals the *neighbor's* corresponding interior
//      pattern (with periodic wrap on periodic axes).
//   3) On non-periodic ends (z walls), halo cells are left at zero
//      (the default Field initial value).
//
// Reports MAX |error| across all interior and halo cells, then
// MPI_Allreduce-reduces and prints from rank 0. Exit code = 0 on success.

#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>

#include "Config.hpp"
#include "Field.hpp"
#include "HaloExchanger.hpp"
#include "MpiTopology.hpp"
#include "Subdomain.hpp"

namespace {

// Encode global indices into a unique double.
double pattern(int I, int J, int K)
{
    return 1.0 * I + 1.0e3 * J + 1.0e6 * K;
}

// Wrap a global index into [1, n] for a periodic axis. For non-periodic
// returns -1 if the index is out of [1,n] (wall ghost — no neighbor data).
int wrap(int g, int n, bool periodic)
{
    if (g >= 1 && g <= n) return g;
    if (!periodic) return -1;
    int r = ((g - 1) % n + n) % n + 1;
    return r;
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 2) {
        if (rank == 0)
            std::fprintf(stderr, "usage: %s <PARA_INPUT.dat>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rc = 0;
    {
        auto cfg = channel::Config::load(argv[1], MPI_COMM_WORLD);
        channel::MpiTopology topo(MPI_COMM_WORLD,
                                  cfg.np1, cfg.np2, cfg.np3,
                                  cfg.pbc1, cfg.pbc2, cfg.pbc3);
        channel::Subdomain sub(cfg, topo);
        channel::HaloExchanger halo(topo, sub);

        channel::Field<double> f(sub.nx(), sub.ny(), sub.nz());
        f.fill(0.0);

        for (int k = 1; k <= sub.nz(); ++k) {
            int K = sub.ista(2) + (k - 1);
            for (int j = 1; j <= sub.ny(); ++j) {
                int J = sub.ista(1) + (j - 1);
                for (int i = 1; i <= sub.nx(); ++i) {
                    int I = sub.ista(0) + (i - 1);
                    f(i, j, k) = pattern(I, J, K);
                }
            }
        }

        halo.exchange(f);

        // Check interior + face ghosts only (corners/edges are not filled
        // by face-only halo exchange — same as PaScaL_TCS).
        double max_err = 0.0;
        long   nbad    = 0;

        bool per[3]  = { cfg.pbc1, cfg.pbc2, cfg.pbc3 };
        int  glob[3] = { cfg.n1m,  cfg.n2m,  cfg.n3m  };
        int  ista[3] = { sub.ista(0), sub.ista(1), sub.ista(2) };
        int  nloc[3] = { sub.nx(),    sub.ny(),    sub.nz()    };

        for (int k = 0; k <= nloc[2] + 1; ++k) {
            for (int j = 0; j <= nloc[1] + 1; ++j) {
                for (int i = 0; i <= nloc[0] + 1; ++i) {
                    int idx[3] = {i, j, k};
                    int n_ghost = 0;
                    for (int a = 0; a < 3; ++a)
                        if (idx[a] == 0 || idx[a] == nloc[a] + 1) ++n_ghost;
                    if (n_ghost > 1) continue;   // skip corners/edges

                    int Kg = ista[2] + (k - 1);
                    int Jg = ista[1] + (j - 1);
                    int Ig = ista[0] + (i - 1);
                    int K  = wrap(Kg, glob[2], per[2]);
                    int J  = wrap(Jg, glob[1], per[1]);
                    int I  = wrap(Ig, glob[0], per[0]);

                    double expected = (I < 0 || J < 0 || K < 0)
                        ? 0.0 : pattern(I, J, K);
                    double err = std::fabs(f(i, j, k) - expected);
                    if (err > max_err) max_err = err;
                    if (err > 1e-12)   ++nbad;
                }
            }
        }

        double max_err_all = 0.0;
        long   nbad_all    = 0;
        MPI_Allreduce(&max_err, &max_err_all, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&nbad,    &nbad_all,    1, MPI_LONG,   MPI_SUM, MPI_COMM_WORLD);

        if (rank == 0) {
            std::printf("[test_halo] max|err|=%g  bad_cells=%ld\n",
                        max_err_all, nbad_all);
            if (max_err_all > 1e-12) {
                std::printf("[test_halo] FAIL\n"); rc = 1;
            } else {
                std::printf("[test_halo] PASS\n");
            }
        }
        MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return rc;
}
