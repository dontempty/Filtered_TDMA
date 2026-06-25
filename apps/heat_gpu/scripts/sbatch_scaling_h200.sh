#!/bin/bash
#SBATCH -J h200_scale
#SBATCH -p amd_h200nv_8
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --comment=inhouse
#SBATCH -o /scratch/x3319a05/Filtered_TDMA/apps/heat_gpu/log/scaling_h200_%j.log
#SBATCH -e /scratch/x3319a05/Filtered_TDMA/apps/heat_gpu/log/scaling_h200_%j.err

echo "=========================================="
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
echo "SLURM_NODELIST = ${SLURM_NODELIST}"
echo "=========================================="

source /usr/share/lmod/lmod/init/bash
module load gcc/11.5.0 mpi/openmpi-4.1.8 nvhpc/25.11_cuda12

PROJ=/scratch/x3319a05/Filtered_TDMA
BIN="${PROJ}/build_sm90/bin/heat_gpu.out"
INP="${PROJ}/apps/heat_gpu/inputs/scaling"
RES="${PROJ}/apps/heat_gpu/results/comm_pure/h200"
mkdir -p "${RES}"

echo "=== [$(date '+%F %T')] Run on $(hostname) ==="
echo "Binary: $(ls -lh ${BIN})"

export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n
MPI_FLAGS="--mca pml ucx --mca osc ucx --mca coll ^hcoll --oversubscribe"
NODE=$(hostname)

run_case() {
    local np=$1 inp_file=$2
    local tag; tag=$(basename "${inp_file}" .txt)
    [ -f "${inp_file}" ] || { echo "  [skip] ${tag}"; return 0; }
    echo "---- np=${np}  [$(date +%H:%M:%S)]  ${tag} ----"
    TIMING_CSV="${RES}/timing_${tag}.csv" \
        mpirun ${MPI_FLAGS} --host ${NODE}:8 -np ${np} "${BIN}" "${inp_file}" 2>&1 \
        | grep -E "rho\]|backend\]|Tmax\]" || true
}

echo ""
echo "===== STRONG SCALING pascal ====="
for rho in rho025 rho040; do for np in 2 4 8; do
    run_case ${np} "${INP}/strong_64x64x512_np${np}_pascal_${rho}.txt"
    run_case ${np} "${INP}/strong_128x128x1024_np${np}_pascal_${rho}.txt"
    run_case ${np} "${INP}/strong_256x256x2048_np${np}_pascal_${rho}.txt"
done; done

echo ""
echo "===== STRONG SCALING filtered_v1 ====="
for rho in rho025 rho040; do for np in 2 4 8; do
    run_case ${np} "${INP}/strong_64x64x512_np${np}_filtered_v1_${rho}.txt"
    run_case ${np} "${INP}/strong_128x128x1024_np${np}_filtered_v1_${rho}.txt"
    run_case ${np} "${INP}/strong_256x256x2048_np${np}_filtered_v1_${rho}.txt"
done; done

echo ""
echo "===== STRONG SCALING filtered_v2 ====="
for rho in rho025 rho040; do for np in 2 4 8; do
    run_case ${np} "${INP}/strong_64x64x512_np${np}_filtered_v2_${rho}.txt"
    run_case ${np} "${INP}/strong_128x128x1024_np${np}_filtered_v2_${rho}.txt"
    run_case ${np} "${INP}/strong_256x256x2048_np${np}_filtered_v2_${rho}.txt"
done; done

echo ""
echo "===== WEAK SCALING pascal ====="
for rho in rho025 rho040; do for np in 2 4 8; do
    run_case ${np} "${INP}/weak_128cube_np${np}_pascal_${rho}.txt"
    run_case ${np} "${INP}/weak_256cube_np${np}_pascal_${rho}.txt"
    run_case ${np} "${INP}/weak_512cube_np${np}_pascal_${rho}.txt"
done; done

echo ""
echo "===== WEAK SCALING filtered_v1 ====="
for rho in rho025 rho040; do for np in 2 4 8; do
    run_case ${np} "${INP}/weak_128cube_np${np}_filtered_v1_${rho}.txt"
    run_case ${np} "${INP}/weak_256cube_np${np}_filtered_v1_${rho}.txt"
    run_case ${np} "${INP}/weak_512cube_np${np}_filtered_v1_${rho}.txt"
done; done

echo ""
echo "===== WEAK SCALING filtered_v2 ====="
for rho in rho025 rho040; do for np in 2 4 8; do
    run_case ${np} "${INP}/weak_128cube_np${np}_filtered_v2_${rho}.txt"
    run_case ${np} "${INP}/weak_256cube_np${np}_filtered_v2_${rho}.txt"
    run_case ${np} "${INP}/weak_512cube_np${np}_filtered_v2_${rho}.txt"
done; done

echo ""
echo "===== REFINE SCALING pascal np=2 ====="
for inp in $(ls "${INP}"/refine_*_np2_pascal_*.txt | sort); do
    run_case 2 "${inp}"
done

echo ""
echo "===== REFINE SCALING filtered_v1 np=2 ====="
for inp in $(ls "${INP}"/refine_*_np2_filtered_v1_*.txt 2>/dev/null | sort); do
    run_case 2 "${inp}"
done

echo ""
echo "===== REFINE SCALING filtered_v2 np=2 ====="
for inp in $(ls "${INP}"/refine_*_np2_filtered_v2_*.txt | sort); do
    run_case 2 "${inp}"
done

echo ""
echo "==== ALL DONE [$(date '+%F %T')]  results: ${RES} ===="
echo "CSVs: $(ls ${RES}/timing_*.csv 2>/dev/null | wc -l)"
