#!/bin/bash
# =============================================================================
#  Heat_gpu STRONG scaling at 1024^3 (separate study), H200 8-GPU node.
#  Grid fixed 1024x1024x1024; z-slab decomposition npz = 1,2,4,8.
#  Backends: pascal + filtered_v2.  rho: 0.25 and 0.40.  (16 runs)
#  Profile timing (filtered routed through *_profile) → tdma_z_{comm,gpu} real.
#  Order: np 8->4->2->1 (light->heavy per GPU) so data flows early.
#
#  Launch:
#    srun --jobid=<JID> --gres=gpu:8 -n1 --overlap \
#         bash apps/heat_gpu/scripts/run_strong_1024_h200.sh
# =============================================================================
set -u
PROJ=/scratch/x3319a05/Filtered_TDMA
source "${PROJ}/env_gpu.sh" 2>/dev/null || true
unset PMIX_INSTALL_PREFIX

BIN="${PROJ}/build_sm90/bin/heat_gpu.out"
INP="${PROJ}/apps/heat_gpu/inputs/scaling"
RES="${PROJ}/apps/heat_gpu/results/scaling_profile/h200/strong_1024"
mkdir -p "${RES}"
[ -x "${BIN}" ] || { echo "ERROR: missing ${BIN}"; exit 1; }

export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n
MPI_FLAGS="--mca pml ucx --mca osc ucx --mca coll ^hcoll --oversubscribe"
NODE=$(hostname)

echo "============================================================"
echo " host : ${NODE}   date : $(date '+%F %T')"
echo " bin  : ${BIN} ($(ls -lh ${BIN} | awk '{print $5}'))"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -8

for np in 8 4 2 1; do
  for be in pascal filtered_v2; do
    for rho in rho025 rho040; do
      inp="${INP}/strong_1024x1024x1024_np${np}_${be}_${rho}.txt"
      tag=$(basename "${inp}" .txt)
      [ -f "${inp}" ] || { echo "[skip missing] ${tag}"; continue; }
      echo ""
      echo "---- ${tag}  NP=${np}  [$(date +%H:%M:%S)] ----"
      T0=$(date +%s)
      TIMING_CSV="${RES}/timing_${tag}.csv" \
        mpirun ${MPI_FLAGS} --host "${NODE}:8" -np "${np}" "${BIN}" "${inp}" 2>&1 \
        | grep -E "rho\]|backend\]|Tmax\]|nx\]|Timing\]|[Ee]rror|fault|illegal|out of memory" || true
      T1=$(date +%s)
      echo "   [wall $((T1-T0))s]  -> timing_${tag}.csv"
    done
  done
done

echo ""; echo "==== DONE $(date '+%F %T') ===="
echo "CSVs: $(ls ${RES}/timing_*.csv 2>/dev/null | wc -l) / 16   in ${RES}"
