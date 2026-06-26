include Makefile.inc

BUILDDIR := build

# ============================================================
#  Top-level orchestration
#
#  Libraries:
#    libs/filtered_tdma/  → build/lib/libfiltered_tdma.a
#    libs/pascal_tdma/    → build/lib/libpascal_tdma.a
#  Apps (pick one with `make channel`, `make channel_gpu`, `make heat`, `make heat_gpu`):
#    apps/channel_cpu/        → build/bin/channel.out
#    apps/heat_cpu/       → build/bin/heat.out
#    apps/heat_gpu/       → build/bin/heat_gpu.out
# ============================================================

.PHONY: all FilteredTDMA PaScaL channel channel_gpu heat heat_gpu tests clean rm

# Output directories produced by a run (stats/, instant/, restart_out/).
# Paths match the defaults in apps/channel_cpu/PARA_INPUT.dat.
RUNDIR := apps/channel_cpu
OUTDIRS := $(RUNDIR)/statistics $(RUNDIR)/instant $(RUNDIR)/restart_out $(RUNDIR)/restart_in

all: FilteredTDMA PaScaL channel

FilteredTDMA: PaScaL
	mkdir -p $(BUILDDIR)/obj/ftdma $(BUILDDIR)/lib $(BUILDDIR)/include
	$(MAKE) -C libs/filtered_tdma all BUILDDIR=../../$(BUILDDIR)

PaScaL:
	mkdir -p $(BUILDDIR)/obj/pascal $(BUILDDIR)/lib $(BUILDDIR)/include
	$(MAKE) -C libs/pascal_tdma all BUILDDIR=../../$(BUILDDIR)

channel: FilteredTDMA PaScaL
	mkdir -p $(BUILDDIR)/obj $(BUILDDIR)/bin
	$(MAKE) -C apps/channel_cpu all BUILDDIR=../../$(BUILDDIR)

channel_gpu: FilteredTDMA PaScaL
	mkdir -p $(BUILDDIR)/obj $(BUILDDIR)/bin
	$(MAKE) -C apps/channel_gpu all BUILDDIR=../../$(BUILDDIR)

heat: FilteredTDMA PaScaL
	mkdir -p $(BUILDDIR)/obj $(BUILDDIR)/bin
	$(MAKE) -C apps/heat_cpu all BUILDDIR=../../$(BUILDDIR)

# GPU heat example: requires USE_CUDA=1 CUDA_ARCH=<sm_*> on the command line.
heat_gpu: FilteredTDMA PaScaL
	mkdir -p $(BUILDDIR)/obj $(BUILDDIR)/bin
	$(MAKE) -C apps/heat_gpu all BUILDDIR=../../$(BUILDDIR)

tests: channel
	mkdir -p $(BUILDDIR)/obj $(BUILDDIR)/bin
	$(MAKE) -C apps/channel_cpu/tests all BUILDDIR=../../../$(BUILDDIR)

clean:
	-$(MAKE) -C libs/filtered_tdma  clean BUILDDIR=../../$(BUILDDIR)
	-$(MAKE) -C libs/pascal_tdma    clean BUILDDIR=../../$(BUILDDIR)
	-$(MAKE) -C apps/channel_cpu        clean BUILDDIR=../../$(BUILDDIR)
	-$(MAKE) -C apps/channel_gpu        clean BUILDDIR=../../$(BUILDDIR) 2>/dev/null
	-$(MAKE) -C apps/heat_cpu       clean BUILDDIR=../../$(BUILDDIR)
	-$(MAKE) -C apps/heat_gpu       clean BUILDDIR=../../$(BUILDDIR) 2>/dev/null
	-$(MAKE) -C apps/channel_cpu/tests  clean BUILDDIR=../../../$(BUILDDIR)
	rm -rf $(BUILDDIR)

# Remove all run-time output (stats/, instant/, restart_out/ under apps/channel_cpu/)
rm:
	@echo "Removing run-time output directories:"
	@for d in $(OUTDIRS); do \
		if [ -e "$$d" ]; then echo "  rm -rf $$d"; rm -rf "$$d"; fi; \
	done
