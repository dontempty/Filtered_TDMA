#include "Config.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace channel {
namespace {

std::string trim(std::string s)
{
    auto issp = [](unsigned char c) { return std::isspace(c); };
    s.erase(s.begin(), std::find_if_not(s.begin(), s.end(), issp));
    s.erase(std::find_if_not(s.rbegin(), s.rend(), issp).base(), s.end());
    return s;
}

bool to_bool(const std::string& v)
{
    std::string s = v;
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (s == "true"  || s == "t" || s == "1" || s == ".true." ) return true;
    if (s == "false" || s == "f" || s == "0" || s == ".false.") return false;
    throw std::runtime_error("Config: cannot parse bool '" + v + "'");
}

using KV = std::unordered_map<std::string, std::string>;

KV parse_file(const std::string& path)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Config: cannot open " + path);

    KV map;
    std::string line;
    while (std::getline(in, line)) {
        // strip inline comment
        auto h = line.find('#');
        if (h != std::string::npos) line.erase(h);
        line = trim(line);
        if (line.empty()) continue;

        auto eq = line.find('=');
        if (eq == std::string::npos) continue;     // skip malformed
        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));
        if (!val.empty() && (val.front() == '"' || val.front() == '\''))
            val = val.substr(1, val.size() - 2);   // strip quotes
        if (!key.empty()) map[key] = val;
    }
    return map;
}

template <typename T> T get(const KV& m, const std::string& k, const T& dflt);

template <> int get<int>(const KV& m, const std::string& k, const int& dflt) {
    auto it = m.find(k); return it == m.end() ? dflt : std::stoi(it->second);
}
template <> double get<double>(const KV& m, const std::string& k, const double& dflt) {
    auto it = m.find(k); return it == m.end() ? dflt : std::stod(it->second);
}
template <> std::string get<std::string>(const KV& m, const std::string& k, const std::string& dflt) {
    auto it = m.find(k); return it == m.end() ? dflt : it->second;
}
template <> bool get<bool>(const KV& m, const std::string& k, const bool& dflt) {
    auto it = m.find(k); return it == m.end() ? dflt : to_bool(it->second);
}

void serialize(const Config& c, std::string& out)
{
    std::ostringstream o;
    o.precision(17);                   // preserve full double precision
    o << std::scientific;
    o << c.n1m << ' ' << c.n2m << ' ' << c.n3m << ' '
      << c.np1 << ' ' << c.np2 << ' ' << c.np3 << ' '
      << (c.pbc1 ? 1 : 0) << ' ' << (c.pbc2 ? 1 : 0) << ' ' << (c.pbc3 ? 1 : 0) << ' '
      << c.uniform1 << ' ' << c.uniform2 << ' ' << c.uniform3 << ' '
      << c.gamma1 << ' ' << c.gamma2 << ' ' << c.gamma3 << ' '
      << c.H << ' ' << c.Aspect1 << ' ' << c.Aspect2 << ' '
      << c.Re_b << ' ' << c.MaxCFL << ' '
      << c.dtStart << ' ' << c.tStart << ' '
      << c.Timestepmax << ' '
      << c.ContinueFilein << ' ' << c.ContinueFileout << ' '
      << static_cast<int>(c.forcing_mode) << ' '
      << c.target_bulk_velocity << ' ' << c.target_dPdx << ' '
      << c.nmonitor << ' '
      << c.nstat_start << ' ' << c.nstat << ' '
      << c.nout_stats << ' ' << c.nout << ' '
      << c.out_stats << ' ' << c.out_field << ' '
      // strings last (length-prefixed)
      << c.dir_cont_filein.size()  << ' ' << c.dir_cont_filein  << ' '
      << c.dir_cont_fileout.size() << ' ' << c.dir_cont_fileout << ' '
      << c.dir_instantfield.size() << ' ' << c.dir_instantfield << ' '
      << c.dir_statistics.size()   << ' ' << c.dir_statistics   << ' '
      << c.tdma_backend.size()     << ' ' << c.tdma_backend;
    out = o.str();
}

void deserialize(const std::string& in, Config& c)
{
    std::istringstream s(in);
    int p1, p2, p3, fmode;
    s >> c.n1m >> c.n2m >> c.n3m
      >> c.np1 >> c.np2 >> c.np3
      >> p1 >> p2 >> p3
      >> c.uniform1 >> c.uniform2 >> c.uniform3
      >> c.gamma1 >> c.gamma2 >> c.gamma3
      >> c.H >> c.Aspect1 >> c.Aspect2
      >> c.Re_b >> c.MaxCFL
      >> c.dtStart >> c.tStart
      >> c.Timestepmax
      >> c.ContinueFilein >> c.ContinueFileout
      >> fmode
      >> c.target_bulk_velocity >> c.target_dPdx
      >> c.nmonitor
      >> c.nstat_start >> c.nstat
      >> c.nout_stats >> c.nout
      >> c.out_stats >> c.out_field;

    c.pbc1 = (p1 != 0); c.pbc2 = (p2 != 0); c.pbc3 = (p3 != 0);
    c.forcing_mode = static_cast<ForcingMode>(fmode);

    auto read_string = [&](std::string& dst) {
        std::size_t len; s >> len; s.get(); // skip space
        dst.resize(len);
        s.read(dst.data(), static_cast<std::streamsize>(len));
    };
    read_string(c.dir_cont_filein);
    read_string(c.dir_cont_fileout);
    read_string(c.dir_instantfield);
    read_string(c.dir_statistics);
    read_string(c.tdma_backend);
}

} // namespace

Config Config::load(const std::string& path, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    Config c;
    std::string blob;
    int ok = 1;

    if (rank == 0) {
        try {
            KV m = parse_file(path);
            c.n1m = get<int>(m, "n1m", 0);
            c.n2m = get<int>(m, "n2m", 0);
            c.n3m = get<int>(m, "n3m", 0);

            c.np1 = get<int>(m, "np1", 1);
            c.np2 = get<int>(m, "np2", 1);
            c.np3 = get<int>(m, "np3", 1);

            c.pbc1 = get<bool>(m, "pbc1", true);
            c.pbc2 = get<bool>(m, "pbc2", true);
            c.pbc3 = get<bool>(m, "pbc3", false);

            c.uniform1 = get<int>(m, "uniform1", 1);
            c.uniform2 = get<int>(m, "uniform2", 1);
            c.uniform3 = get<int>(m, "uniform3", 0);

            c.gamma1 = get<double>(m, "gamma1", 0.0);
            c.gamma2 = get<double>(m, "gamma2", 0.0);
            c.gamma3 = get<double>(m, "gamma3", 0.0);

            c.H       = get<double>(m, "H",       2.0);
            c.Aspect1 = get<double>(m, "Aspect1", 4.0);
            c.Aspect2 = get<double>(m, "Aspect2", 2.0);

            c.Re_b   = get<double>(m, "Re_b",   2800.0);
            c.MaxCFL = get<double>(m, "MaxCFL", 1.0);

            c.dtStart             = get<double>(m, "dtStart", 1.0e-3);
            c.tStart              = get<double>(m, "tStart",  0.0);
            c.Timestepmax         = get<int>(m, "Timestepmax", 10000);
            c.ContinueFilein  = get<int>(m, "ContinueFilein",  0);
            c.ContinueFileout = get<int>(m, "ContinueFileout", 1);
            c.dir_cont_filein  = get<std::string>(m, "dir_cont_filein",  "./restart_in/");
            c.dir_cont_fileout = get<std::string>(m, "dir_cont_fileout", "./restart_out/");
            c.dir_instantfield = get<std::string>(m, "dir_instantfield", "./instant/");
            c.dir_statistics   = get<std::string>(m, "dir_statistics",   "./statistics/");

            std::string fm = get<std::string>(m, "forcing_mode", "MASS_FLOW");
            std::transform(fm.begin(), fm.end(), fm.begin(),
                           [](unsigned char ch){ return std::toupper(ch); });
            c.forcing_mode = (fm == "PRESSURE_GRADIENT")
                ? ForcingMode::PRESSURE_GRADIENT : ForcingMode::MASS_FLOW;
            c.target_bulk_velocity = get<double>(m, "target_bulk_velocity", 1.0);
            c.target_dPdx          = get<double>(m, "target_dPdx", 0.0);

            c.nmonitor    = get<int>(m, "nmonitor",    1);
            c.nstat_start = get<int>(m, "nstat_start", 0);
            c.nstat       = get<int>(m, "nstat",       1);
            c.nout_stats  = get<int>(m, "nout_stats",  1000);
            c.nout        = get<int>(m, "nout",        10000);
            c.out_stats   = get<int>(m, "out_stats",   1);
            c.out_field   = get<int>(m, "out_field",   1);

            c.tdma_backend = get<std::string>(m, "tdma_backend", "filtered");

            // derived
            c.n1 = c.n1m + 1;
            c.n2 = c.n2m + 1;
            c.n3 = c.n3m + 1;
            c.Lx = c.H * c.Aspect1;
            c.Ly = c.H * c.Aspect2;
            c.Lz = c.H;

            serialize(c, blob);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[Config] %s\n", e.what());
            ok = 0;
        }
    }

    MPI_Bcast(&ok, 1, MPI_INT, 0, comm);
    if (!ok) MPI_Abort(comm, 1);

    int len = (rank == 0) ? static_cast<int>(blob.size()) : 0;
    MPI_Bcast(&len, 1, MPI_INT, 0, comm);
    if (rank != 0) blob.resize(len);
    MPI_Bcast(blob.data(), len, MPI_CHAR, 0, comm);
    if (rank != 0) deserialize(blob, c);

    // re-derive on non-root (cheap, avoids transmitting them)
    c.n1 = c.n1m + 1; c.n2 = c.n2m + 1; c.n3 = c.n3m + 1;
    c.Lx = c.H * c.Aspect1; c.Ly = c.H * c.Aspect2; c.Lz = c.H;

    return c;
}

void Config::print() const
{
    std::printf("====== Channel Config ======\n");
    std::printf("  meshes:    n1m=%d n2m=%d n3m=%d (n1=%d n2=%d n3=%d)\n",
                n1m, n2m, n3m, n1, n2, n3);
    std::printf("  MPI_procs: np1=%d np2=%d np3=%d (total=%d)\n",
                np1, np2, np3, np1*np2*np3);
    std::printf("  periodic:  pbc1=%d pbc2=%d pbc3=%d (z is wall-normal)\n",
                pbc1, pbc2, pbc3);
    std::printf("  uniform:   %d %d %d   stretch gamma: %g %g %g\n",
                uniform1, uniform2, uniform3, gamma1, gamma2, gamma3);
    std::printf("  domain:    Lx=%g Ly=%g Lz=%g (H=%g, Aspect1=%g Aspect2=%g)\n",
                Lx, Ly, Lz, H, Aspect1, Aspect2);
    std::printf("  flow:      Re_b=%g MaxCFL=%g\n", Re_b, MaxCFL);
    std::printf("  time:      dtStart=%g tStart=%g Timestepmax=%d\n",
                dtStart, tStart, Timestepmax);
    const char* fm = (forcing_mode == ForcingMode::MASS_FLOW)
        ? "MASS_FLOW" : "PRESSURE_GRADIENT";
    std::printf("  forcing:   %s  Ub_target=%g  dPdx_target=%g\n",
                fm, target_bulk_velocity, target_dPdx);
    std::printf("  restart:   in=%d out=%d  '%s' '%s'\n",
                ContinueFilein, ContinueFileout,
                dir_cont_filein.c_str(), dir_cont_fileout.c_str());
    std::printf("  output:    nmonitor=%d  nstat_start=%d  nstat=%d\n",
                nmonitor, nstat_start, nstat);
    std::printf("             nout_stats=%d  nout=%d  out_stats=%d  out_field=%d\n",
                nout_stats, nout, out_stats, out_field);
    std::printf("             stat_dir='%s' instant_dir='%s'\n",
                dir_statistics.c_str(), dir_instantfield.c_str());
    std::printf("  tdma:      backend=%s\n", tdma_backend.c_str());
    std::printf("============================\n");
}

} // namespace channel
