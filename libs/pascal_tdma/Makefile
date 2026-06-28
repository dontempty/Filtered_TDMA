include ../../Makefile.inc

BUILDDIR ?= ../../build

LIB    = libpascal_tdma.a
OBJDIR = $(BUILDDIR)/obj/pascal
LIBDIR = $(BUILDDIR)/lib
INCDIR = $(BUILDDIR)/include

# CPU sources (cpu/*.cpp)
CPP_SOURCES = $(wildcard cpu/*.cpp)
CPP_OBJS    = $(patsubst cpu/%.cpp, $(OBJDIR)/%.o, $(CPP_SOURCES))

# GPU sources (gpu/*.cu) — only when USE_CUDA=1
ifeq ($(USE_CUDA),1)
    CU_SOURCES = $(wildcard gpu/*.cu)
    CU_OBJS    = $(patsubst gpu/%.cu, $(OBJDIR)/%.o, $(CU_SOURCES))
endif

OBJS = $(CPP_OBJS) $(CU_OBJS)

all: $(LIBDIR)/$(LIB) inc

$(LIBDIR)/$(LIB): $(OBJS)
	@echo "Creating static library: $@"
	ar qc $@ $^

inc:
	@echo "Copying public headers to $(INCDIR)/"
	cp include/*.hpp $(INCDIR)/

$(OBJDIR)/%.o: cpu/%.cpp | $(OBJDIR)
	@echo "Compiling $< -> $@"
	$(CXX) $(CXXFLAGS) -c $< -o $@ -MMD

$(OBJDIR)/%.o: gpu/%.cu | $(OBJDIR)
	@echo "Compiling $< -> $@"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR) $(LIBDIR) $(INCDIR)

clean:
	@echo "Cleaning..."
	rm -f $(OBJS) $(CPP_OBJS:.o=.d) $(LIBDIR)/$(LIB)
	rm -f $(INCDIR)/pascal_tdma_many.hpp $(INCDIR)/pascal_tdma_many_cuda.hpp

-include $(CPP_OBJS:.o=.d)
