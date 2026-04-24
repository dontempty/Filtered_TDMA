include ../Makefile.inc

BUILDDIR ?= ../build

LIB    = libfiltered_tdma.a
SRCDIR = .
OBJDIR = $(BUILDDIR)/obj/ftdma
LIBDIR = $(BUILDDIR)/lib
INCDIR = $(BUILDDIR)/include

# Exclude tdma_local and para_range — provided by libpascal_tdma (superset version)
SOURCES = $(filter-out $(SRCDIR)/tdma_local.cpp $(SRCDIR)/para_range.cpp, \
           $(wildcard $(SRCDIR)/*.cpp))
OBJS    = $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SOURCES))

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

$(OBJDIR):
	mkdir -p $(OBJDIR) $(LIBDIR) $(INCDIR)

clean:
	@echo "Cleaning src..."
	rm -f $(OBJS) $(OBJS:.o=.d) $(LIBDIR)/$(LIB)
	rm -f $(INCDIR)/*.hpp

-include $(OBJS:.o=.d)
