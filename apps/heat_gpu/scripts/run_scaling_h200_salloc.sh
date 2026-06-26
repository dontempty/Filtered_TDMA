#!/bin/bash
# Interactive H200 scaling: build (sm_90) + strong/weak sweep on the allocated
# node. Modules are loaded on the LAUNCHING side (nvhpc/25.11_cuda12, which
# provides nvcc + hpcx mpicxx + hpcx mpirun) and propagated by srun --export=ALL;
# this script does NOT run `module` (compute nodes use Tcl Modules and can't
# reload the Lmod nvhpc reliably). Launch via:
#   source /etc/profile.d/lmod.sh; module purge; module load nvhpc/25.11_cuda12
#   srun --jobid=<JID> --gres=gpu:8 -n1 --overlap bash run_scaling_h200_salloc.sh
set -u
PROJ=/scratch/x3319a05/Filtered_TDMA
BD="${PROJ}/build_sm90"
BIN="${BD}/bin/heat_gpu.out"
INP="${PROJ}/apps/heat_gpu/inputs/scaling"
RES="${PROJ}/apps/heat_gpu/results/scaling_1comm/h200"
mkdir -p "${RES}"

echo "=== env (inherited from launcher) on $(hostname)  [$(date '+%F %T')] ==="
echo "  nvcc:   $(which nvcc 2>&1)"
echo "  mpicxx: $(which mpicxx 2>&1)"
echo "  mpirun: $(which mpirun 2>&1)"
command -v nvcc  >/dev/null || { echo "ERROR: nvcc not in PATH — load nvhpc before srun"; exit 1; }
command -v mpirun >/dev/null || { echo "ERROR: mpirun not in PATH"; exit 1; }

# ---- build (relative BUILDDIR; tools from inherited nvhpc) ----
echo "=== [BUILD] heat_gpu sm_90 → ${BD} ==="
( cd "${PROJ}" && rm -rf "${BD}" && USE_CUDA=1 CUDA_ARCH=90 make heat_gpu BUILDDIR=build_sm90 ) 2>&1 | tail -4
[ -x "${BIN}" ] || { echo "ERROR: build failed — ${BIN} missing"; exit 1; }
echo "[build ok] $(ls -lh ${BIN} | awk '{print $5, $NF}')"

# ---- runtime env ----
export CUDA_LIBDIR="${NVHPC_ROOT:-}/cuda/lib64"
export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n
MPI_FLAGS="--mca pml ucx --mca osc ucx --mca coll ^hcoll --oversubscribe"
NODE=$(hostname)
echo "=== GPUs on ${NODE} ==="; nvidia-smi --query-gpu=index,name --format=csv,noheader | head -8

run_case() {
    local label=$1 np=$2 inp_file=$3
    local tag; tag=$(basename "${inp_file}" .txt)
    [ -f "${inp_file}" ] || { echo "  [skip missing] ${tag}"; return 0; }
    echo "---- ${label}  np=${np}  [$(date +%H:%M:%S)]  ${tag} ----"
    TIMING_CSV="${RES}/timing_${tag}.csv" \
        mpirun ${MPI_FLAGS} --host "${NODE}:8" -np ${np} "${BIN}" "${inp_file}" 2>&1 \
        | grep -E "rho\]|backend\]|Tmax\]|ERROR|error|fault" || true
    return 0
}

echo ""; echo "===== STRONG SCALING  np=2,4,8 ====="
for rho in rho025 rho040; do for be in pascal filtered_v2; do for np in 2 4 8; do
    run_case "strong 64x64x512"    ${np} "${INP}/strong_64x64x512_np${np}_${be}_${rho}.txt"
    run_case "strong 128x128x1024" ${np} "${INP}/strong_128x128x1024_np${np}_${be}_${rho}.txt"
    run_case "strong 256x256x2048" ${np} "${INP}/strong_256x256x2048_np${np}_${be}_${rho}.txt"
done; done; done

echo ""; echo "===== WEAK SCALING  np=2,4,8 ====="
for rho in rho025 rho040; do for be in pascal filtered_v2; do for np in 2 4 8; do
    run_case "weak 128cube/GPU" ${np} "${INP}/weak_128cube_np${np}_${be}_${rho}.txt"
    run_case "weak 256cube/GPU" ${np} "${INP}/weak_256cube_np${np}_${be}_${rho}.txt"
    run_case "weak 512cube/GPU" ${np} "${INP}/weak_512cube_np${np}_${be}_${rho}.txt"
done; done; done

echo ""; echo "==== ALL DONE [$(date '+%F %T')]  results: ${RES} ===="
echo "CSV files: $(ls ${RES}/*.csv 2>/dev/null | wc -l)"
