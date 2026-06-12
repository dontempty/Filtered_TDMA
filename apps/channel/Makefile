include ../../Makefile.inc

# ============================================================
#  Channel Flow Solver — Makefile
#  Requires:  mpicxx (OpenMPI/MPICH)
#             FFTW3 at $(FFTW_DIR)
#             FilteredTDMA lib from ../build/
# ============================================================

BUILDDIR ?= ../../build

EXE      := channel.out
SRCDIR   := .
OBJDIR   := $(BUILDDIR)/obj
IDIR     := $(BUILDDIR)/include
LDIR     := $(BUILDDIR)/lib
BINDIR   := $(BUILDDIR)/bin

SOURCES  := $(wildcard $(SRCDIR)/*.cpp)
OBJS     := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/ch_%.o, $(SOURCES))

# Extra includes: FFTW3 and FilteredTDMA
CXXFLAGS += -I$(FFTW_DIR)/include -I$(IDIR)

# Link: FFTW3 (serial only — we drive 1D plans manually), FilteredTDMA + PaScaL_TDMA libs
LDFLAGS  := -L$(LDIR) -L$(FFTW_DIR)/lib
LIBS     := -lfiltered_tdma -lpascal_tdma -lfftw3 -lm

# Rpath so the binary finds FFTW at runtime
LDFLAGS  += -Wl,-rpath,$(FFTW_DIR)/lib

all: dirs $(BINDIR)/$(EXE)

dirs:
	mkdir -p $(OBJDIR) $(BINDIR)

$(BINDIR)/$(EXE): $(OBJS)
	@echo "Linking: $@"
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

$(OBJDIR)/ch_%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	@echo "Compiling $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@ -MMD

-include $(OBJS:.o=.d)

clean:
	rm -f $(OBJS) $(OBJS:.o=.d) $(BINDIR)/$(EXE)

.PHONY: all dirs clean
