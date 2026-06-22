#!/bin/bash
#SBATCH --job-name=v100_opt
#SBATCH --partition=cas_v100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --comment=inhouse
#SBATCH --output=/scratch/x3319a05/Filtered_TDMA/apps/heat_gpu/log/scaling_v100_opt.out
#SBATCH --error=/scratch/x3319a05/Filtered_TDMA/apps/heat_gpu/log/scaling_v100_opt.err

PROJ=/scratch/x3319a05/Filtered_TDMA
BIN="${PROJ}/build/bin/heat_gpu.out"
INP="${PROJ}/apps/heat_gpu/inputs/scaling"
RES="${PROJ}/apps/heat_gpu/results/scaling_gpu_opt"

echo "=== START $(date '+%F %T') ==="
echo "Node: $(hostname)  JOB=${SLURM_JOB_ID}"

module purge
module load nvhpc/25.11_cuda12

export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n
MPI_FLAGS="--mca pml ucx --mca osc ucx --mca coll ^hcoll --oversubscribe"

NODE=$(hostname)
nvidia-smi -L
echo "Binary: $(ls -lh ${BIN})"

run_case() {
    local label=$1 np=$2 inp_file=$3
    local tag; tag=$(basename "${inp_file}" .txt)
    echo "---- ${label}  np=${np}  [$(date +%H:%M:%S)] ----"
    TIMING_CSV="${RES}/timing_${tag}.csv" \
        mpirun ${MPI_FLAGS} --host "${NODE}:4" -np ${np} "${BIN}" "${inp_file}" 2>&1
    [ $? -ne 0 ] && echo "[ERROR]"
    return 0
}

echo ""; echo "=== STRONG np=1,2,4 ==="
for rho in rho025 rho040; do for backend in pascal filtered_v2; do
    for np in 1 2 4; do
        run_case "strong 64²×512"   ${np} "${INP}/strong_64x64x512_np${np}_${backend}_${rho}.txt"
        run_case "strong 128²×1024" ${np} "${INP}/strong_128x128x1024_np${np}_${backend}_${rho}.txt"
        run_case "strong 256²×2048" ${np} "${INP}/strong_256x256x2048_np${np}_${backend}_${rho}.txt"
    done
done; done

echo ""; echo "=== WEAK np=1,2,4 ==="
for rho in rho025 rho040; do for backend in pascal filtered_v2; do
    for np in 1 2 4; do
        run_case "weak 128³/GPU" ${np} "${INP}/weak_128cube_np${np}_${backend}_${rho}.txt"
        run_case "weak 256³/GPU" ${np} "${INP}/weak_256cube_np${np}_${backend}_${rho}.txt"
        run_case "weak 512³/GPU" ${np} "${INP}/weak_512cube_np${np}_${backend}_${rho}.txt"
    done
done; done

echo ""; echo "=== REFINEMENT np=2 ==="
for rho in rho025 rho040; do for backend in pascal filtered_v2; do
    for nz in 64 128 256; do
        run_case "refine 64×64×${nz}"   2 "${INP}/refine_64x64x${nz}_np2_${backend}_${rho}.txt"
    done
    for nz in 128 256 512; do
        run_case "refine 128×128×${nz}" 2 "${INP}/refine_128x128x${nz}_np2_${backend}_${rho}.txt"
    done
    for nz in 256 512 1024; do
        run_case "refine 256×256×${nz}" 2 "${INP}/refine_256x256x${nz}_np2_${backend}_${rho}.txt"
    done
done; done

echo ""; echo "==== ALL DONE [$(date '+%F %T')] ===="
