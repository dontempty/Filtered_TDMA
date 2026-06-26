include ../../Makefile.inc

BUILDDIR ?= ../../build

EXE      := channel_gpu.out
SRCDIR   := .
OBJDIR   := $(BUILDDIR)/obj
IDIR     := $(BUILDDIR)/include
LDIR     := $(BUILDDIR)/lib
BINDIR   := $(BUILDDIR)/bin

CPP_SOURCES := Config.cpp FieldOutput.cpp Grid.cpp MpiTopology.cpp RestartIO.cpp Subdomain.cpp TdmaSolverGPU.cpp main.cpp
CU_SOURCES  := BoundaryConditionGPU.cu ChannelForcingGPU.cu DeviceField.cu HaloExchangerGPU.cu MomentumSolverGPU.cu PressureSolverGPU.cu StatisticsGPU.cu TimeIntegratorGPU.cu
CPP_OBJS    := $(patsubst %.cpp,$(OBJDIR)/chgpu_%.o,$(CPP_SOURCES))
CU_OBJS     := $(patsubst %.cu,$(OBJDIR)/chgpu_%.o,$(CU_SOURCES))
OBJS        := $(CPP_OBJS) $(CU_OBJS)

CUDA_LIBDIR ?= $(if $(CUDA_HOME),$(CUDA_HOME)/lib64,/usr/local/cuda/lib64)
CUDA_INCDIR ?= $(CUDA_LIBDIR:/lib64=/include)
CUDA_MATHLIBDIR ?= $(if $(NVHPC_ROOT),$(NVHPC_ROOT)/math_libs/12.9/targets/x86_64-linux/lib,$(CUDA_LIBDIR))

CXXFLAGS += -I$(IDIR) -I$(CUDA_INCDIR)
LDFLAGS  := -L$(LDIR) -L$(CUDA_LIBDIR) -L$(CUDA_MATHLIBDIR)
LDFLAGS  += -Wl,-rpath,$(CUDA_LIBDIR) -Wl,-rpath,$(CUDA_MATHLIBDIR)
LIBS     := -lfiltered_tdma -lpascal_tdma $(CUDA_LDFLAGS) -lcufft -lstdc++ -lm

all: dirs $(BINDIR)/$(EXE)

dirs:
	mkdir -p $(OBJDIR) $(BINDIR)

$(BINDIR)/$(EXE): $(OBJS)
	@echo "Linking: $@"
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

$(OBJDIR)/chgpu_%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	@echo "Compiling $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@ -MMD

$(OBJDIR)/chgpu_%.o: $(SRCDIR)/%.cu | $(OBJDIR)
	@echo "Compiling $<"
	$(NVCC) $(NVCCFLAGS) -I$(IDIR) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	rm -f $(OBJS) $(CPP_OBJS:.o=.d) $(BINDIR)/$(EXE)

-include $(CPP_OBJS:.o=.d)

.PHONY: all dirs clean
