#ifndef TIMING_CSV_HPP
#define TIMING_CSV_HPP
//
// C++ port of PaScaL_TDMA_F/examples/debug_utils.f90 — per-timestep,
// per-rank, per-event timing accumulator + long-format CSV writer.
//
// Usage:
//   timing_init(n_events, n_steps, comm);                  // before time loop
//   timing_record(t_step, local_times, comm);              // every timestep
//   timing_save_csv(path, event_names, meta_header, comm); // after loop
//   timing_cleanup();
//
// Output:
//   # <meta header line>
//   rank,t_step,event,time_sec
//   0,1,rhs, 1.234567890E-04
//   ...
//

#include <cstdio>
#include <mpi.h>
#include <string>
#include <vector>

namespace timing_csv {

inline std::vector<double>& _buffer() { static std::vector<double> b; return b; }
inline int& _n_events() { static int v = 0; return v; }
inline int& _n_ranks () { static int v = 0; return v; }
inline int& _n_steps () { static int v = 0; return v; }

inline void timing_init(int n_events, int n_steps, MPI_Comm comm) {
    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);
    _n_events() = n_events;
    _n_ranks () = nprocs;
    _n_steps () = n_steps;
    if (myrank == 0) {
        _buffer().assign((std::size_t)n_events * nprocs * n_steps, 0.0);
    } else {
        _buffer().clear();
    }
}

inline void timing_record(int t_step,
                          const std::vector<double>& local_times,
                          MPI_Comm comm) {
    int myrank;
    MPI_Comm_rank(comm, &myrank);
    int n_events = _n_events();
    int n_ranks  = _n_ranks();

    std::vector<double> gather_buf;
    double* recv = nullptr;
    if (myrank == 0) {
        gather_buf.resize((std::size_t)n_events * n_ranks);
        recv = gather_buf.data();
    }
    MPI_Gather(local_times.data(), n_events, MPI_DOUBLE,
               recv,               n_events, MPI_DOUBLE,
               0, comm);

    if (myrank == 0) {
        // t_step is 0-based (caller passes t_step - n_warmup, first timed step = 0).
        // timing_save_csv reads slots 0..n_steps-1, so write to slot t_step directly.
        std::size_t base = (std::size_t)n_events * n_ranks * (std::size_t)t_step;
        for (int r = 0; r < n_ranks; ++r)
            for (int e = 0; e < n_events; ++e)
                _buffer()[base + e + (std::size_t)n_events * r]
                    = gather_buf[(std::size_t)r * n_events + e];
    }
}

inline void timing_save_csv(const std::string& filename,
                            const std::vector<std::string>& event_names,
                            const std::string& meta_header,
                            MPI_Comm comm) {
    int myrank;
    MPI_Comm_rank(comm, &myrank);
    if (myrank != 0) return;

    std::FILE* f = std::fopen(filename.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "[Timing] cannot open %s for write\n", filename.c_str());
        return;
    }
    std::fprintf(f, "# %s\n", meta_header.c_str());
    std::fprintf(f, "rank,t_step,event,time_sec\n");

    int n_events = _n_events();
    int n_ranks  = _n_ranks();
    int n_steps  = _n_steps();
    for (int r = 0; r < n_ranks; ++r) {
        for (int t = 1; t <= n_steps; ++t) {
            std::size_t base = (std::size_t)n_events * n_ranks * (std::size_t)(t - 1)
                             + (std::size_t)n_events * r;
            for (int e = 0; e < n_events; ++e) {
                std::fprintf(f, "%d,%d,%s,%16.9E\n",
                             r, t, event_names[e].c_str(),
                             _buffer()[base + e]);
            }
        }
    }
    std::fclose(f);
    std::printf(" [Timing] CSV saved to %s\n", filename.c_str());
}

inline void timing_cleanup() {
    _buffer().clear();
    _buffer().shrink_to_fit();
    _n_events() = 0;
    _n_ranks () = 0;
    _n_steps () = 0;
}

} // namespace timing_csv

#endif // TIMING_CSV_HPP
