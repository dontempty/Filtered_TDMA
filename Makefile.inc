# ============================================================
#  Channel — Compiler / flag configuration
# ============================================================

# ---------- g++ ----------
# CXX      := mpicxx
# CXXFLAGS := -O3 -std=c++17 -g -fPIC -march=native
# CXXFLAGS += -fno-fast-math -ffp-contract=off

# ---------- nvhpc ----------
# CXX      := mpicxx
# CXXFLAGS := -O3 -std=c++17 -fPIC -tp=icelake -fast

# ---------- g++ ----------
CXX      := mpicxx
CXXFLAGS := -O3 -std=c++17 -fPIC -march=native

# ---------- intel ----------
# CXX       := mpiicpx
# CXXFLAGS  := -O3 -std=c++17 -g -fPIC -fp-model precise -march=native

# ---------- External library paths ----------
FFTW_DIR := /shared/home/wel1come1234/local/fftw3
