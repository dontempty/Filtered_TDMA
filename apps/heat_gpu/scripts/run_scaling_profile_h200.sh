#!/bin/bash
# =============================================================================
#  Heat_gpu scaling re-run with the *profile* TDMA solve (H200, 8-GPU node).
#
#  Drives strong / weak / refine sweeps off apps/heat_gpu/inputs/scaling/*.txt,
#  restricted to the two backends we care about: pascal and filtered_v2.
#
#  Why "profile": TdmaBackendGPU::solve() now routes the filtered backend through
#  solve_filtered_v2_profile(), which populates last_comm_ms()/last_gpu_ms() so
#  the per-direction tdma_{z,y,x}_{comm,gpu} CSV events are real (the sync-free
#  production solve left them as stale zeros). PASCAL already times itself.
#
#  Launch from the login node into the live allocation, e.g.:
#    srun --jobid=794488 --gres=gpu:8 -n1 --overlap \
#         bash apps/heat_gpu/scripts/run_scaling_profile_h200.sh
# =============================================================================
set -u

PROJ=/scratch/x3319a05/Filtered_TDMA
# Self-contained toolchain (nvhpc-only; provides mpirun + cuda libs on the node).
source "${PROJ}/env_gpu.sh" 2>/dev/null || true
# module load nvhpc sets PMIX_INSTALL_PREFIX, which conflicts with the hpcx
# OMPI's internal PMIx when launched under srun. Drop it so mpirun starts.
unset PMIX_INSTALL_PREFIX
BIN="${PROJ}/build_sm90/bin/heat_gpu.out"
INP="${PROJ}/apps/heat_gpu/inputs/scaling"
RESBASE="${PROJ}/apps/heat_gpu/results/scaling_profile/h200"
mkdir -p "${RESBASE}/strong" "${RESBASE}/weak" "${RESBASE}/refine"

[ -x "${BIN}" ] || { echo "ERROR: missing ${BIN}"; exit 1; }

# CUDA-aware MPI / UCX (matches the working refine_h200_salloc.sh flow).
export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n
MPI_FLAGS="--mca pml ucx --mca osc ucx --mca coll ^hcoll --oversubscribe"
NODE=$(hostname)

echo "============================================================"
echo " host : ${NODE}"
echo " date : $(date '+%F %T')"
echo " bin  : ${BIN} ($(ls -lh ${BIN} | awk '{print $5}'))"
echo "============================================================"
nvidia-smi --query-gpu=index,name --format=csv,noheader | head -8
command -v mpirun >/dev/null || { echo "ERROR: mpirun not in PATH"; exit 1; }

# Run every input matching <prefix-glob> for backends {pascal, filtered_v2}.
#   $1 = category dir (strong|weak|refine)
#   $2 = glob with the literal token BE where the backend name goes
run_set () {
    local category="$1"; local glob="$2"
    local res="${RESBASE}/${category}"
    echo ""; echo "################## ${category} ##################"
    for be in pascal filtered_v2; do
        local pat="${glob/BE/${be}}"
        for inp in $(ls ${INP}/${pat} 2>/dev/null | sort -V); do
            local tag NP T0 T1
            tag=$(basename "${inp}" .txt)
            NP=$(awk '/^npx /{x=$3}/^npy /{y=$3}/^npz /{z=$3} END{print x*y*z}' "${inp}")
            echo "---- ${tag}  NP=${NP}  [$(date +%H:%M:%S)] ----"
            T0=$(date +%s)
            TIMING_CSV="${res}/timing_${tag}.csv" \
                mpirun ${MPI_FLAGS} --host "${NODE}:8" -np "${NP}" "${BIN}" "${inp}" 2>&1 \
                | grep -E "rho\]|backend\]|Tmax\]|L2|Timing\]|ERROR|[Ee]rror|fault|illegal" || true
            T1=$(date +%s)
            echo "   [wall $((T1-T0))s]  -> timing_${tag}.csv"
        done
    done
}

run_set strong "strong_*_np*_BE_rho*.txt"
run_set weak   "weak_*_np*_BE_rho*.txt"
run_set refine "refine_*_np2_BE_rho*.txt"

echo ""; echo "==== DONE $(date '+%F %T') ===="
for c in strong weak refine; do
    echo "  ${c}: $(ls ${RESBASE}/${c}/timing_*.csv 2>/dev/null | wc -l) CSVs"
done
