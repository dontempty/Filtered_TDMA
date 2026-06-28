#!/bin/bash
source /etc/profile.d/modules.sh
module load nvhpc/25.11_cuda12 >/dev/null 2>&1
cd /scratch/x3319a05/Filtered_TDMA || exit 99
rm -rf build
echo "### PASCAL ###"
make -C libs/pascal_tdma   BUILDDIR=../../build
echo "### FILTERED (rc from make) ###"
make -C libs/filtered_tdma BUILDDIR=../../build
echo "### HEAT_CPU ###"
make -C apps/heat_cpu      BUILDDIR=../../build
echo "### DONE ###"
