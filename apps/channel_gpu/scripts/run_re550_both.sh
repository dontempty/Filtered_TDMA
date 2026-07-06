#!/bin/bash
# Run channel_gpu Re550 with 2x2x2 = 8 GPUs.
# Runs pascal first, then filtered (sequentially — each uses all 8 GPUs).
#
# Launch from login node into the live allocation:
#   ssh gpu56 "bash /scratch/x3319a05/Filtered_TDMA/apps/channel_gpu/scripts/run_re550_both.sh"
set -u

PROJ=/scratch/x3319a05/Filtered_TDMA
BIN="${PROJ}/build/bin/channel_gpu.out"
WDIR="${PROJ}/apps/channel_gpu"
NODE=$(hostname)

echo "=== channel_gpu Re550 (2x2x2 = 8 GPUs) on ${NODE}  [$(date '+%F %T')] ==="
[ -x "${BIN}" ] || { echo "ERROR: ${BIN} missing"; exit 1; }

source "${PROJ}/env_gpu.sh" 2>/dev/null || true
unset PMIX_INSTALL_PREFIX OPAL_PREFIX

export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n
MPI_FLAGS="--bind-to none --mca pml ucx --mca osc ucx --mca coll ^hcoll --oversubscribe"

cd "${WDIR}"
mkdir -p log \
         statistics/re550_p statistics/re550_f \
         instant/re550_p    instant/re550_f \
         restart_out/re550_p restart_out/re550_f \
         restart_in/re550_p  restart_in/re550_f

# ---- [1] Pascal ----
echo ""
echo "=== [1] Pascal  (8 GPUs, 2x2x2) ==="
T0=$(date +%s)
mpirun ${MPI_FLAGS} --host "${NODE}:8" -np 8 \
    "${BIN}" input/PARA_INPUT_re550_pascal.dat \
    > log/re550_pascal.log 2>&1
RC_P=$?
T1=$(date +%s)
echo "  rc=${RC_P}  wall=$((T1-T0))s"
tail -5 log/re550_pascal.log

# ---- [2] Filtered ----
echo ""
echo "=== [2] Filtered  (8 GPUs, 2x2x2) ==="
T0=$(date +%s)
mpirun ${MPI_FLAGS} --host "${NODE}:8" -np 8 \
    "${BIN}" input/PARA_INPUT_re550_filtered.dat \
    > log/re550_filtered.log 2>&1
RC_F=$?
T1=$(date +%s)
echo "  rc=${RC_F}  wall=$((T1-T0))s"
tail -5 log/re550_filtered.log

echo ""
echo "==== ALL DONE [$(date '+%F %T')] ===="
echo "  pascal   rc=${RC_P}   log: log/re550_pascal.log"
echo "  filtered rc=${RC_F}   log: log/re550_filtered.log"
