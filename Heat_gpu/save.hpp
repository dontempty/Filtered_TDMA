#ifndef SAVE_HPP
#define SAVE_HPP

#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

/// Save a 3D field (flat array, k-major) as one CSV per z-slice.
inline void save_3d_to_csv(const std::vector<double>& arr,
                           int nx, int ny, int nz,
                           const std::string& folder,
                           const std::string& base_filename,
                           int precision = 6)
{
    namespace fs = std::filesystem;
    fs::create_directories(folder);

    for (int k = 0; k < nz; ++k) {
        std::string fname = base_filename + "_k" + std::to_string(k) + ".csv";
        std::ofstream ofs((fs::path(folder) / fname).string());
        if (!ofs) { std::cerr << "Error: cannot open " << fname << "\n"; continue; }
        ofs << std::fixed << std::setprecision(precision);
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                ofs << arr[k * ny * nx + j * nx + i];
                if (i < nx - 1) ofs << ',';
            }
            ofs << '\n';
        }
    }
}

#endif // SAVE_HPP
