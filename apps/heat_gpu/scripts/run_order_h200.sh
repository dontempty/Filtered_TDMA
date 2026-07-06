#!/bin/bash
# =============================================================================
#  Order-of-accuracy verification for heat_gpu on H200.
#
#  Tests all 3 backends (pascal, filtered, filtered_v2) x rho (0.25, 0.40)
#  x np (1, 2, 4) using 32^3 grids with option=order.
#
#  Inputs : apps/heat_gpu/inputs/order_accuracy/
#  Results: printed to stdout (L2 error per case)
#
#  Launch from the login node into the live allocation, e.g.:
#    ssh gpu56 "bash /scratch/x3319a05/Filtered_TDMA/apps/heat_gpu/scripts/run_order_h200.sh"
# =============================================================================
set -u

PROJ=/scratch/x3319a05/Filtered_TDMA
source "${PROJ}/env_gpu.sh" 2>/dev/null || true
unset PMIX_INSTALL_PREFIX OPAL_PREFIX

BIN="${PROJ}/build/bin/heat_gpu.out"
INP="${PROJ}/apps/heat_gpu/inputs/order_accuracy"

[ -x "${BIN}" ] || { echo "ERROR: missing ${BIN}"; exit 1; }

export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n
MPI_FLAGS="--mca pml ucx --mca osc ucx --mca coll ^hcoll --oversubscribe"
NODE=$(hostname)

echo "============================================================"
echo " host : ${NODE}"
echo " date : $(date '+%F %T')"
echo " bin  : ${BIN}"
echo "============================================================"

for rho_str in rho025 rho040; do
    echo ""
    echo "################## ${rho_str} ##################"
    for backend in pascal filtered filtered_v2; do
        echo "--- ${backend} ---"
        for np in 1 2 4; do
            inp="${INP}/in_${backend}_${rho_str}_np${np}.txt"
            [ -f "${inp}" ] || { echo "  np=${np}: [skip missing ${inp}]"; continue; }
            echo -n "  np=${np}: "
            mpirun ${MPI_FLAGS} --host "${NODE}:8" -np ${np} "${BIN}" "${inp}" 2>/dev/null \
                | grep "L2 error"
        done
    done
done

echo ""
echo "==== DONE $(date '+%F %T') ===="
