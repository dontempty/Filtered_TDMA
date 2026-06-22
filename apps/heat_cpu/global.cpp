#include "global.hpp"
#include <fstream>
#include <sstream>
#include <map>
#include <cmath>
#include <algorithm>
#include <cctype>
#include <stdexcept>

static inline void trim(std::string& s) {
    auto f = [](char c) { return !std::isspace(static_cast<unsigned char>(c)); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), f));
    s.erase(std::find_if(s.rbegin(), s.rend(), f).base(), s.end());
}

void GlobalParams::load(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile) throw std::runtime_error("Cannot open input file: " + filename);

    std::map<std::string, std::string> param;
    std::string line;
    while (std::getline(infile, line)) {
        if (auto pos = line.find('!'); pos != std::string::npos) line.erase(pos);
        trim(line);
        if (line.empty()) continue;

        std::string key, value;
        if (auto eq = line.find('='); eq != std::string::npos) {
            key   = line.substr(0, eq);
            value = line.substr(eq + 1);
        } else {
            std::istringstream iss(line);
            if (!(iss >> key >> value)) continue;
        }
        trim(key); trim(value);
        if (!key.empty() && !value.empty()) param[key] = value;
    }

    nx           = std::stoi(param.at("nx"));
    ny           = std::stoi(param.at("ny"));
    nz           = std::stoi(param.at("nz"));
    np_dim[0]    = std::stoi(param.at("npx"));
    np_dim[1]    = std::stoi(param.at("npy"));
    np_dim[2]    = std::stoi(param.at("npz"));
    rho          = std::stod(param.at("rho"));
    eps_constant = std::stod(param.at("eps"));
    Tmax         = param.count("Tmax")   ? std::stod(param.at("Tmax"))   : 0.0;
    dt           = param.count("dt")     ? std::stod(param.at("dt"))     : 0.0;
    Nt           = param.count("Nt")     ? std::stoi(param.at("Nt"))     : 0;
    Nt_warmup    = param.count("warmup") ? std::stoi(param.at("warmup")) : 0;
    option       = param.at("option");
    if (auto it = param.find("tdma_backend"); it != param.end()) {
        tdma_backend = it->second;
    }

    // Grid dimensions (add 1 for node-centered)
    nx++; ny++; nz++;
    nxm = nx - 1; nym = ny - 1; nzm = nz - 1;

    // Domain extents
    x0 = -1.0; xN = 1.0;
    y0 = -1.0; yN = 1.0;
    z0 = -1.0; zN = 1.0;
    lx = xN - x0; ly = yN - y0; lz = zN - z0;
    dx = lx / (nx - 1);
    dy = ly / (ny - 1);
    dz = lz / (nz - 1);

    if (option == "order") {
        // dt from rho and dx (spatial convergence: uniform grid, dx ∝ 1/N)
        dt   = rho / (1.0 - 2.0 * rho) * (2.0 * dx * dx);
        // Reference Tmax: 128 steps at N=512
        double dt_N = rho / (1.0 - 2.0 * rho) * (2.0 * (lx / 512.0) * (lx / 512.0));
        Tmax = dt_N * 128.0;
        Nt   = static_cast<int>(std::round(Tmax / dt));
    } else if (option == "strong") {
        // dt from rho and dz so that rho_z is fixed regardless of nz.
        // Nt (total steps incl. warmup) and warmup are read from input.
        dt   = rho / (1.0 - 2.0 * rho) * (2.0 * dz * dz);
        Tmax = Nt * dt;
    } else {
        dt   = rho / (1.0 - 2.0 * rho) * (2.0 * dx * dx);
        double dt_N = rho / (1.0 - 2.0 * rho) * (2.0 * (lx / 512.0) * (lx / 512.0));
        Tmax = dt_N * 128.0;
        Nt   = static_cast<int>(std::round(Tmax / dt));
    }
}
