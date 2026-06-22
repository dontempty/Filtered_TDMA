#!/bin/bash
# V100에서 filtered_v2 order-of-accuracy 테스트
# salloc 노드에서 실행: bash run_order_v100.sh

set -e
PROJ=$(realpath ../..)
BIN="${PROJ}/build/bin/heat_gpu.out"
INP_DIR="${PROJ}/apps/heat_gpu/inputs"

module purge
module load nvhpc/25.11_cuda12

# Build (sm_70 = V100)
echo "==== Building heat_gpu (sm_70) ===="
cd "${PROJ}"
make clean >/dev/null 2>&1 || true
USE_CUDA=1 CUDA_ARCH=70 make heat_gpu 2>&1 | tail -5

echo ""
echo "==== Order-of-Accuracy: filtered_v2 (1 GPU) ===="
echo "  N    L2_error"
for N in 64 128 256 512; do
    INP="${INP_DIR}/PARA_INPUT_1gpu_${N}.txt"
    OUT=$(mpirun -np 1 "${BIN}" "${INP}" 2>&1)
    ERR=$(echo "${OUT}" | grep -i "L2\|error\|norm" | tail -1)
    printf "  %-4d  %s\n" "${N}" "${ERR}"
done
