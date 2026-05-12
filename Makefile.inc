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
# CXX      := mpicxx
# CXXFLAGS := -O3 -std=c++17 -fPIC -march=native

# ---------- intel ----------
# CXX       := mpiicpx
# CXXFLAGS  := -O3 -std=c++17 -g -fPIC -fp-model precise -march=native

# ---------- Kisti (g++) ----------
CXX      := mpicxx
CXXFLAGS := -O3 -std=c++17 -fPIC -march=x86-64-v3

# ---------- External library paths ----------
FFTW_DIR := /apps/libraries/fftw3/3.3.10/gcc/11.5.0

# ============================================================
#  CUDA / GPU build (set USE_CUDA=1 to build the GPU variant)
#  KISTI Neuron: module load nvhpc/25.11_cuda12
# ============================================================
USE_CUDA  ?= 0

# nvcc; -ccbin uses the MPI C++ wrapper so MPI headers / link are consistent.
NVCC      := nvcc
# Override as needed (70 V100, 80 A100, 90 H100).
CUDA_ARCH ?= 80
NVCCFLAGS := -O3 -std=c++17 -Xcompiler -fPIC -arch=sm_$(CUDA_ARCH) -ccbin $(CXX)
CUDA_LDFLAGS := -lcudart
# nvhpc bundles libcudart at $(NVHPC_ROOT)/cuda/lib64 — heat_gpu Makefile
# uses CUDA_LIBDIR to find it.
CUDA_LIBDIR ?= $(if $(NVHPC_ROOT),$(NVHPC_ROOT)/cuda/lib64,$(if $(CUDA_HOME),$(CUDA_HOME)/lib64,/usr/local/cuda/lib64))
