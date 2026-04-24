#include "RestartIO.hpp"

#include "MpiTopology.hpp"
#include "Subdomain.hpp"

#include <cstdio>
#include <cstring>
#include <mpi.h>
#include <vector>

namespace channel {

RestartIO::RestartIO(const MpiTopology& topo, const Subdomain& sub)
    : topo_(topo), sub_(sub) {}

namespace {

inline std::size_t gidx(int I, int J, int K, int Nx, int Ny) {
    return static_cast<std::size_t>(I)
         + static_cast<std::size_t>(Nx)
           * (static_cast<std::size_t>(J)
              + static_cast<std::size_t>(Ny) * static_cast<std::size_t>(K));
}

void write_blob(const std::string& path, const void* p, std::size_t bytes)
{
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    std::fwrite(p, 1, bytes, f);
    std::fclose(f);
}
void read_blob(const std::string& path, void* p, std::size_t bytes)
{
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return;
    std::fread(p, 1, bytes, f);
    std::fclose(f);
}

} // namespace

void RestartIO::gather_field_(const Field<double>& f, double* global_buf) const
{
    const int nx = sub_.nx(), ny = sub_.ny(), nz = sub_.nz();
    const int Nx_g = sub_.global_n(0);
    const int Ny_g = sub_.global_n(1);
    const int Nz_g = sub_.global_n(2);
    const int rank = topo_.rank();
    const int nprocs = topo_.nprocs();

    // Each rank packs its interior into a contiguous send buffer.
    std::vector<double> sendbuf(static_cast<std::size_t>(nx) * ny * nz);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                sendbuf[i + nx * (j + ny * k)] = f(i + 1, j + 1, k + 1);

    // Gather sizes and offsets to rank 0; gather the actual data.
    int local_n = nx * ny * nz;
    std::vector<int> rcounts(nprocs, 0), rdisps(nprocs, 0);
    std::vector<int> isendsz(6, 0);
    isendsz[0] = sub_.ista(0); isendsz[1] = sub_.iend(0);
    isendsz[2] = sub_.ista(1); isendsz[3] = sub_.iend(1);
    isendsz[4] = sub_.ista(2); isendsz[5] = sub_.iend(2);

    std::vector<int> isizes(6 * nprocs, 0);
    MPI_Gather(isendsz.data(), 6, MPI_INT,
               isizes.data(),  6, MPI_INT, 0, topo_.cart());
    MPI_Gather(&local_n, 1, MPI_INT, rcounts.data(), 1, MPI_INT, 0, topo_.cart());

    if (rank == 0) {
        rdisps[0] = 0;
        for (int r = 1; r < nprocs; ++r) rdisps[r] = rdisps[r - 1] + rcounts[r - 1];
    }
    std::vector<double> recvbuf;
    if (rank == 0) recvbuf.assign(static_cast<std::size_t>(Nx_g) * Ny_g * Nz_g, 0.0);

    // Use MPI_Gatherv into a flat receive buffer; rank 0 then unpacks.
    std::vector<double> flat;
    if (rank == 0) flat.resize(static_cast<std::size_t>(Nx_g) * Ny_g * Nz_g);
    MPI_Gatherv(sendbuf.data(), local_n, MPI_DOUBLE,
                flat.data(), rcounts.data(), rdisps.data(), MPI_DOUBLE,
                0, topo_.cart());

    if (rank == 0) {
        for (int r = 0; r < nprocs; ++r) {
            int ist = isizes[6*r + 0], ien = isizes[6*r + 1];
            int jst = isizes[6*r + 2], jen = isizes[6*r + 3];
            int kst = isizes[6*r + 4], ken = isizes[6*r + 5];
            int rnx = ien - ist + 1;
            int rny = jen - jst + 1;
            int rnz = ken - kst + 1;
            int off = rdisps[r];
            for (int k = 0; k < rnz; ++k)
                for (int j = 0; j < rny; ++j)
                    for (int i = 0; i < rnx; ++i) {
                        int Ig = ist + i - 1;
                        int Jg = jst + j - 1;
                        int Kg = kst + k - 1;
                        global_buf[gidx(Ig, Jg, Kg, Nx_g, Ny_g)] =
                            flat[off + i + rnx * (j + rny * k)];
                    }
        }
    }
}

void RestartIO::scatter_field_(const double* global_buf, Field<double>& f) const
{
    const int nx = sub_.nx(), ny = sub_.ny(), nz = sub_.nz();
    const int Nx_g = sub_.global_n(0);
    const int Ny_g = sub_.global_n(1);
    const int Nz_g = sub_.global_n(2);
    const int rank = topo_.rank();
    const int nprocs = topo_.nprocs();

    int local_n = nx * ny * nz;
    std::vector<int> rcounts(nprocs, 0), rdisps(nprocs, 0);
    int isendsz[6] = { sub_.ista(0), sub_.iend(0),
                       sub_.ista(1), sub_.iend(1),
                       sub_.ista(2), sub_.iend(2) };
    std::vector<int> isizes(6 * nprocs, 0);
    MPI_Gather(isendsz, 6, MPI_INT, isizes.data(), 6, MPI_INT, 0, topo_.cart());
    MPI_Gather(&local_n, 1, MPI_INT, rcounts.data(), 1, MPI_INT, 0, topo_.cart());

    std::vector<double> flat;
    if (rank == 0) {
        flat.assign(static_cast<std::size_t>(Nx_g) * Ny_g * Nz_g, 0.0);
        rdisps[0] = 0;
        for (int r = 1; r < nprocs; ++r) rdisps[r] = rdisps[r - 1] + rcounts[r - 1];
        for (int r = 0; r < nprocs; ++r) {
            int ist = isizes[6*r + 0], ien = isizes[6*r + 1];
            int jst = isizes[6*r + 2], jen = isizes[6*r + 3];
            int kst = isizes[6*r + 4], ken = isizes[6*r + 5];
            int rnx = ien - ist + 1;
            int rny = jen - jst + 1;
            int rnz = ken - kst + 1;
            int off = rdisps[r];
            for (int k = 0; k < rnz; ++k)
                for (int j = 0; j < rny; ++j)
                    for (int i = 0; i < rnx; ++i) {
                        int Ig = ist + i - 1;
                        int Jg = jst + j - 1;
                        int Kg = kst + k - 1;
                        flat[off + i + rnx * (j + rny * k)] =
                            global_buf[gidx(Ig, Jg, Kg, Nx_g, Ny_g)];
                    }
        }
        (void)Nz_g;
    }

    std::vector<double> recvbuf(local_n, 0.0);
    MPI_Scatterv(flat.data(), rcounts.data(), rdisps.data(), MPI_DOUBLE,
                 recvbuf.data(), local_n, MPI_DOUBLE, 0, topo_.cart());

    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                f(i + 1, j + 1, k + 1) = recvbuf[i + nx * (j + ny * k)];
}

void RestartIO::write(const std::string& dir,
                      const Field<double>& U, const Field<double>& V,
                      const Field<double>& W, const Field<double>& P,
                      const RestartState& s)
{
    const int Nx_g = sub_.global_n(0);
    const int Ny_g = sub_.global_n(1);
    const int Nz_g = sub_.global_n(2);
    const std::size_t N = static_cast<std::size_t>(Nx_g) * Ny_g * Nz_g;

    std::vector<double> buf;
    if (topo_.rank() == 0) buf.assign(N, 0.0);

    auto dump = [&](const Field<double>& f, const std::string& name) {
        gather_field_(f, buf.data());
        if (topo_.rank() == 0)
            write_blob(dir + "/" + name + ".bin", buf.data(), N * sizeof(double));
    };
    dump(U, "cont_U");
    dump(V, "cont_V");
    dump(W, "cont_W");
    dump(P, "cont_P");

    if (topo_.rank() == 0) {
        double meta[4] = {s.time, s.dt, static_cast<double>(s.step), s.dPdx};
        write_blob(dir + "/cont_time.bin", meta, sizeof(meta));
    }
}

void RestartIO::read(const std::string& dir,
                     Field<double>& U, Field<double>& V,
                     Field<double>& W, Field<double>& P,
                     RestartState& s)
{
    const int Nx_g = sub_.global_n(0);
    const int Ny_g = sub_.global_n(1);
    const int Nz_g = sub_.global_n(2);
    const std::size_t N = static_cast<std::size_t>(Nx_g) * Ny_g * Nz_g;

    std::vector<double> buf;
    if (topo_.rank() == 0) buf.assign(N, 0.0);

    auto load = [&](Field<double>& f, const std::string& name) {
        if (topo_.rank() == 0)
            read_blob(dir + "/" + name + ".bin", buf.data(), N * sizeof(double));
        scatter_field_(buf.data(), f);
    };
    load(U, "cont_U");
    load(V, "cont_V");
    load(W, "cont_W");
    load(P, "cont_P");

    double meta[4] = {0, 0, 0, 0};
    if (topo_.rank() == 0)
        read_blob(dir + "/cont_time.bin", meta, sizeof(meta));
    MPI_Bcast(meta, 4, MPI_DOUBLE, 0, topo_.cart());
    s.time = meta[0]; s.dt = meta[1]; s.step = static_cast<long>(meta[2]);
    s.dPdx = meta[3];
}

} // namespace channel
