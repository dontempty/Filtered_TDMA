include ../../../Makefile.inc

# ============================================================
#  Channel — unit tests (built against the channel example)
# ============================================================

BUILDDIR ?= ../../../build

OBJDIR  := $(BUILDDIR)/obj
IDIR    := $(BUILDDIR)/include
LDIR    := $(BUILDDIR)/lib
BINDIR  := $(BUILDDIR)/bin

CXXFLAGS += -I$(FFTW_DIR)/include -I$(IDIR) -I..
LDFLAGS  := -L$(LDIR) -L$(FFTW_DIR)/lib
LIBS     := -lfiltered_tdma -lfftw3 -lm
LDFLAGS  += -Wl,-rpath,$(FFTW_DIR)/lib

# Channel-side objects required by the tests (provided by the channel build,
# so `make channel` from the top level must run first).
CH_OBJS := $(OBJDIR)/ch_Config.o \
           $(OBJDIR)/ch_MpiTopology.o \
           $(OBJDIR)/ch_Subdomain.o \
           $(OBJDIR)/ch_Grid.o \
           $(OBJDIR)/ch_HaloExchanger.o

TESTS  := test_halo
EXES   := $(addprefix $(BINDIR)/, $(addsuffix .out, $(TESTS)))

all: dirs $(EXES)

dirs:
	mkdir -p $(OBJDIR) $(BINDIR)

$(BINDIR)/test_halo.out: $(OBJDIR)/test_halo.o $(CH_OBJS)
	@echo "Linking $@"
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

$(OBJDIR)/%.o: %.cpp | $(OBJDIR)
	@echo "Compiling $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@ -MMD

-include $(OBJDIR)/test_halo.d

clean:
	rm -f $(OBJDIR)/test_*.o $(OBJDIR)/test_*.d $(EXES)

run: all
	@for t in $(TESTS); do \
	  echo "== running $$t =="; \
	  mpirun -np 1 --mca btl ^openib $(BINDIR)/$$t.out ../PARA_INPUT.dat || exit 1; \
	done

.PHONY: all dirs clean run
