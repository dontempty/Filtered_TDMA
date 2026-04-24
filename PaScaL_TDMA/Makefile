include ../Makefile.inc

BUILDDIR ?= ../build

LIB    = libpascal_tdma.a
SRCDIR = .
OBJDIR = $(BUILDDIR)/obj/pascal
LIBDIR = $(BUILDDIR)/lib
INCDIR = $(BUILDDIR)/include

SOURCES = $(wildcard $(SRCDIR)/*.cpp)
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
