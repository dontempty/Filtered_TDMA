include ../../Makefile.inc

BUILDDIR ?= ../../build

EXE    := heat_gpu.out
SRCDIR := .
OBJDIR := $(BUILDDIR)/obj
IDIR   := $(BUILDDIR)/include
LDIR   := $(BUILDDIR)/lib
BINDIR := $(BUILDDIR)/bin

CPP_SOURCES := $(wildcard $(SRCDIR)/*.cpp)
CU_SOURCES  := $(wildcard $(SRCDIR)/*.cu)
CPP_OBJS    := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/exgpu_%.o, $(CPP_SOURCES))
CU_OBJS     := $(patsubst $(SRCDIR)/%.cu,  $(OBJDIR)/exgpu_%.o, $(CU_SOURCES))
OBJS        := $(CPP_OBJS) $(CU_OBJS)

LDFLAGS := -L$(LDIR)
# pascal_tdma static archive contains CUDA objects, so we need the CUDA runtime.
CUDA_LIBDIR ?= $(if $(CUDA_HOME),$(CUDA_HOME)/lib64,/usr/local/cuda/lib64)
CUDA_INCDIR ?= $(CUDA_LIBDIR:/lib64=/include)
LIBS    := -lfiltered_tdma -lpascal_tdma -L$(CUDA_LIBDIR) $(CUDA_LDFLAGS) -lstdc++
CXXFLAGS += -I$(IDIR) -I$(CUDA_INCDIR)

all: $(BINDIR)/$(EXE)

$(BINDIR)/$(EXE): $(OBJS)
	@echo "Linking: $@"
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

$(OBJDIR)/exgpu_%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	@echo "Compiling $< -> $@"
	$(CXX) $(CXXFLAGS) -c $< -o $@ -MMD

$(OBJDIR)/exgpu_%.o: $(SRCDIR)/%.cu | $(OBJDIR)
	@echo "Compiling $< -> $@"
	$(NVCC) $(NVCCFLAGS) -I$(IDIR) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	@echo "Cleaning heat_gpu..."
	rm -f $(OBJS) $(CPP_OBJS:.o=.d) $(BINDIR)/$(EXE)

-include $(CPP_OBJS:.o=.d)
