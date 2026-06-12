include ../../Makefile.inc

BUILDDIR ?= ../../build

LIB    = libpascal_tdma.a
SRCDIR = .
OBJDIR = $(BUILDDIR)/obj/pascal
LIBDIR = $(BUILDDIR)/lib
INCDIR = $(BUILDDIR)/include

CPP_SOURCES = $(wildcard $(SRCDIR)/*.cpp)
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
ifeq ($(USE_CUDA),1)
	cp *.cuh $(INCDIR)/
endif

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	@echo "Compiling $< -> $@"
	$(CXX) $(CXXFLAGS) -c $< -o $@ -MMD

$(OBJDIR)/%.o: $(SRCDIR)/%.cu | $(OBJDIR)
	@echo "Compiling $< -> $@"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR) $(LIBDIR) $(INCDIR)

clean:
	@echo "Cleaning src..."
	rm -f $(OBJS) $(CPP_OBJS:.o=.d) $(LIBDIR)/$(LIB)
	rm -f $(INCDIR)/*.hpp $(INCDIR)/*.cuh

-include $(CPP_OBJS:.o=.d)
