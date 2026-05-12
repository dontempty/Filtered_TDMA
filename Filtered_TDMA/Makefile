include ../Makefile.inc

BUILDDIR ?= ../build

LIB    = libfiltered_tdma.a
SRCDIR = .
OBJDIR = $(BUILDDIR)/obj/ftdma
LIBDIR = $(BUILDDIR)/lib
INCDIR = $(BUILDDIR)/include

# Exclude tdma_local and para_range — provided by libpascal_tdma (superset version)
CPP_SOURCES = $(filter-out $(SRCDIR)/tdma_local.cpp $(SRCDIR)/para_range.cpp, \
              $(wildcard $(SRCDIR)/*.cpp))
CPP_OBJS    = $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(CPP_SOURCES))

# Add CUDA sources only when USE_CUDA=1
ifeq ($(USE_CUDA),1)
    CU_SOURCES = $(wildcard $(SRCDIR)/*.cu)
    CU_OBJS    = $(patsubst $(SRCDIR)/%.cu, $(OBJDIR)/%.o, $(CU_SOURCES))
endif

OBJS = $(CPP_OBJS) $(CU_OBJS)

all: $(LIBDIR)/$(LIB) inc

$(LIBDIR)/$(LIB): $(OBJS)
	@echo "Creating static library: $@"
	ar qc $@ $^

inc:
	@echo "Copying header files to $(INCDIR)/"
	cp *.hpp $(INCDIR)/

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	@echo "Compiling $< -> $@"
	$(CXX) $(CXXFLAGS) -c $< -o $@ -MMD

$(OBJDIR)/%.o: $(SRCDIR)/%.cu | $(OBJDIR)
	@echo "Compiling $< -> $@"
	$(NVCC) $(NVCCFLAGS) -I$(INCDIR) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR) $(LIBDIR) $(INCDIR)

clean:
	@echo "Cleaning src..."
	rm -f $(OBJS) $(CPP_OBJS:.o=.d) $(LIBDIR)/$(LIB)
	rm -f $(INCDIR)/*.hpp

-include $(CPP_OBJS:.o=.d)
