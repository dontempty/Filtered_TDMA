include Makefile.inc

BUILDDIR := build

# ============================================================
#  Top-level orchestration
#
#  Libraries:
#    Filtered_TDMA/  → build/lib/libfiltered_tdma.a
#    PaScaL_TDMA/    → build/lib/libpascal_tdma.a
#  Solvers (pick one with `make channel` or `make heat`):
#    channel/        → build/bin/channel.out
#    Heat/           → build/bin/heat.out
# ============================================================

.PHONY: all FilteredTDMA PaScaL channel heat heat_gpu tests clean rm

# Output directories produced by a run (stats/, instant/, restart_out/).
# Paths match the defaults in channel/PARA_INPUT.dat.
RUNDIR := channel
OUTDIRS := $(RUNDIR)/statistics $(RUNDIR)/instant $(RUNDIR)/restart_out $(RUNDIR)/restart_in

all: FilteredTDMA PaScaL channel

FilteredTDMA: PaScaL
	mkdir -p $(BUILDDIR)/obj/ftdma $(BUILDDIR)/lib $(BUILDDIR)/include
	$(MAKE) -C Filtered_TDMA all BUILDDIR=../$(BUILDDIR)

PaScaL:
	mkdir -p $(BUILDDIR)/obj/pascal $(BUILDDIR)/lib $(BUILDDIR)/include
	$(MAKE) -C PaScaL_TDMA all BUILDDIR=../$(BUILDDIR)

channel: FilteredTDMA PaScaL
	mkdir -p $(BUILDDIR)/obj $(BUILDDIR)/bin
	$(MAKE) -C channel all BUILDDIR=../$(BUILDDIR)

heat: FilteredTDMA PaScaL
	mkdir -p $(BUILDDIR)/obj $(BUILDDIR)/bin
	$(MAKE) -C Heat all BUILDDIR=../$(BUILDDIR)

# GPU heat example: requires USE_CUDA=1 CUDA_ARCH=<sm_*> on the command line.
heat_gpu: FilteredTDMA PaScaL
	mkdir -p $(BUILDDIR)/obj $(BUILDDIR)/bin
	$(MAKE) -C Heat_gpu all BUILDDIR=../$(BUILDDIR)

tests: channel
	mkdir -p $(BUILDDIR)/obj $(BUILDDIR)/bin
	$(MAKE) -C channel/tests all BUILDDIR=../../$(BUILDDIR)

clean:
	-$(MAKE) -C Filtered_TDMA  clean BUILDDIR=../$(BUILDDIR)
	-$(MAKE) -C PaScaL_TDMA    clean BUILDDIR=../$(BUILDDIR)
	-$(MAKE) -C channel        clean BUILDDIR=../$(BUILDDIR)
	-$(MAKE) -C Heat           clean BUILDDIR=../$(BUILDDIR)
	-$(MAKE) -C Heat_gpu       clean BUILDDIR=../$(BUILDDIR) 2>/dev/null
	-$(MAKE) -C channel/tests  clean BUILDDIR=../../$(BUILDDIR)
	rm -rf $(BUILDDIR)

# Remove all run-time output (stats/, instant/, restart_out/ under channel/)
rm:
	@echo "Removing run-time output directories:"
	@for d in $(OUTDIRS); do \
		if [ -e "$$d" ]; then echo "  rm -rf $$d"; rm -rf "$$d"; fi; \
	done
