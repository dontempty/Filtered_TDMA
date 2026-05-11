#ifndef DEBUG_HPP
#define DEBUG_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <mpi.h>

/// Gather per-rank timing data to rank 0 and write to a text file.
inline void save_timing_data(const std::string& filename, MPI_Comm comm,
                             const std::vector<std::string>& event_names,
                             const std::vector<double>& local_times)
{
    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    std::vector<double> all_times;
    if (myrank == 0) all_times.resize(nprocs * local_times.size());

    MPI_Gather(local_times.data(), local_times.size(), MPI_DOUBLE,
               all_times.data(), local_times.size(), MPI_DOUBLE, 0, comm);

    if (myrank == 0) {
        std::ofstream ofs(filename);
        if (!ofs) { std::cerr << "Error: cannot open " << filename << "\n"; return; }
        ofs << std::fixed << std::setprecision(9);
        for (int rank = 0; rank < nprocs; ++rank)
            for (size_t e = 0; e < event_names.size(); ++e)
                ofs << "[" << event_names[e] << "]: "
                    << all_times[rank * event_names.size() + e] << "\n";
        std::cout << "Timing data saved to " << filename << "\n";
    }
}

#endif // DEBUG_HPP
