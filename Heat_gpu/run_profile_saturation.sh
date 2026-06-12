#!/bin/bash
#SBATCH -J ftdma_prof
#SBATCH -p cas_v100_2
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH -o log/%x_%j.out
#SBATCH -e log/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --comment etc

# ============================================================
#  GPU saturation analysis + per-kernel overhead breakdown.
#
#  Task 1 (saturation): ncu measures DRAM throughput and SM utilization
#                       across N=512 NP=1 / N=1024 NP=2 / N=1024 NP=8.
#                       If GPU is bandwidth-saturated at N=1024 but not
#                       at N=512 → confirms the "saturation point" hypothesis.
#
#  Task 2 (overhead):   nsys gets per-kernel timing breakdown.
#                       Same per-rank workload (134M cells) at NP=1 vs NP=8
#                       lets us see exactly where the 35% extra time goes
#                       (boundary kernels? halo? TDMA itself?).
# ============================================================
set +e

module purge
module load nvhpc/25.11_cuda12

export UCX_MEMTYPE_CACHE=n
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=^openib,uct

PROJ="/scratch/x3319a05/Filtered_TDMA"
EXE="${PROJ}/build/bin/heat_gpu.out"
RESULTS="${PROJ}/Heat_gpu/profile_results"
mkdir -p "${RESULTS}" log
NCU=/usr/local/cuda-13.0/bin/ncu
NSYS=/usr/local/cuda-13.0/bin/nsys

[ -x "${EXE}" ] || { echo "[ERROR] missing ${EXE}" >&2; exit 1; }
[ -x "${NCU}" ] || { echo "[ERROR] missing ${NCU}" >&2; exit 1; }
[ -x "${NSYS}" ] || { echo "[ERROR] missing ${NSYS}" >&2; exit 1; }

echo "============================================================"
echo " host : $(hostname)"
echo " date : $(date '+%F %T')"
echo " gpus : $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "============================================================"

# ---- Configs: tag, input_path, NP ----
# Reduced to 2 GPU max (cas_v100_2 partition).
# Covers three workload-per-rank points to map the saturation curve:
#   N=512 NP=2  → 67M cells/rank  (under-saturated hypothesis)
#   N=512 NP=1  → 134M cells/rank (transition)
#   N=1024 NP=2 → 537M cells/rank (saturated hypothesis)
declare -a CONFIGS=(
  "N512_NP2   inputs/strong_512/PARA_INPUT_2.txt   2"
  "N512_NP1   inputs/strong_512/PARA_INPUT_1.txt   1"
  "N1024_NP2  inputs/strong_1024/PARA_INPUT_2.txt  2"
)

# =================================================================
# Task 2: nsys per-kernel timing breakdown (rank 0 only via wrapping)
# Light overhead — runs at near-normal speed.
# =================================================================
echo ""
echo "######  TASK 2 — nsys per-kernel breakdown  ######"
for entry in "${CONFIGS[@]}"; do
  set -- $entry; TAG=$1; INP=${PROJ}/Heat_gpu/$2; NP=$3
  echo ""
  echo "=== nsys ${TAG} ==="
  REPORT="${RESULTS}/nsys_${TAG}"
  # nsys-aware MPI launch (rank 0 only profiled to avoid trace bloat)
  mpirun --bind-to none -np ${NP} bash -c '
    if [ "${OMPI_COMM_WORLD_RANK}" = "0" ]; then
      exec '"${NSYS}"' profile -f true -o '"${REPORT}"' \
            --trace=cuda,mpi --sample=none --cpuctxsw=none \
            '"${EXE}"' '"${INP}"'
    else
      exec '"${EXE}"' '"${INP}"'
    fi
  ' 2>&1
  echo "[nsys done] report: ${REPORT}.nsys-rep"
done

# =================================================================
# Task 1: ncu DRAM throughput + SM utilization
# Heavier overhead — profile 2 launches per kernel after skipping 3.
# Focus on tdma_many_kernel (filtered TDMA), rhs_kernel, build_lhs_z_kernel.
# =================================================================
echo ""
echo "######  TASK 1 — ncu DRAM/SM saturation  ######"
METRICS="dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
launch__occupancy_limit_active_warps,\
launch__registers_per_thread,\
launch__block_size,\
launch__grid_size,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum"

KERNEL_REGEX="tdma_many_kernel|modified_thomas_kernel|rhs_kernel|build_lhs_z_kernel|build_lhs_y_kernel|build_lhs_x_kernel"

for entry in "${CONFIGS[@]}"; do
  set -- $entry; TAG=$1; INP=${PROJ}/Heat_gpu/$2; NP=$3
  echo ""
  echo "=== ncu ${TAG} ==="
  CSV="${RESULTS}/ncu_${TAG}.csv"
  # ncu rank 0 only, skip first 3 launches to avoid warmup, sample 2 per kernel.
  mpirun --bind-to none -np ${NP} bash -c '
    if [ "${OMPI_COMM_WORLD_RANK}" = "0" ]; then
      exec '"${NCU}"' --target-processes=application-only \
            --launch-skip 3 --launch-count 2 \
            --kernel-name regex:"'"${KERNEL_REGEX}"'" \
            --metrics '"${METRICS}"' \
            --csv --log-file '"${CSV}"' \
            '"${EXE}"' '"${INP}"'
    else
      exec '"${EXE}"' '"${INP}"'
    fi
  ' 2>&1 | tail -20
  echo "[ncu done] csv: ${CSV}"
done

echo ""
echo "######  ALL DONE  ######"
ls -la "${RESULTS}/"
