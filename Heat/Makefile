include ../Makefile.inc

# ============================================================
#  Heat Equation Example — Makefile
#  Requires: mpicxx
#            FilteredTDMA + PaScaL_TDMA libs from ../build/
# ============================================================

BUILDDIR ?= ../build

EXE      := heat.out
SRCDIR   := .
OBJDIR   := $(BUILDDIR)/obj
IDIR     := $(BUILDDIR)/include
LDIR     := $(BUILDDIR)/lib
BINDIR   := $(BUILDDIR)/bin

SOURCES  := $(wildcard $(SRCDIR)/*.cpp)
OBJS     := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/ht_%.o, $(SOURCES))

CXXFLAGS += -I$(IDIR)

LDFLAGS  := -L$(LDIR)
LIBS     := -lfiltered_tdma -lpascal_tdma -lm

all: dirs $(BINDIR)/$(EXE)

dirs:
	mkdir -p $(OBJDIR) $(BINDIR)

$(BINDIR)/$(EXE): $(OBJS)
	@echo "Linking: $@"
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

$(OBJDIR)/ht_%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	@echo "Compiling $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@ -MMD

-include $(OBJS:.o=.d)

clean:
	rm -f $(OBJS) $(OBJS:.o=.d) $(BINDIR)/$(EXE)

.PHONY: all dirs clean
