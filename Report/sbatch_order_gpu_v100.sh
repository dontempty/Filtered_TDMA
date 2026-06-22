#!/bin/bash
#SBATCH -J ghostGPU
#SBATCH -p cas_v100_2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/x3319a05/Filtered_TDMA/Report/order_gpu_v100_%j.out
#SBATCH -e /scratch/x3319a05/Filtered_TDMA/Report/order_gpu_v100_%j.out
#SBATCH --time=00:20:00
#SBATCH --comment etc
set -u
PROJ=/scratch/x3319a05/Filtered_TDMA
BIN="${PROJ}/build/bin/heat_gpu.out"

source /usr/share/Modules/init/bash 2>/dev/null || source /etc/profile.d/lmod.sh 2>/dev/null || true
module purge >/dev/null 2>&1
module load nvhpc/25.11_cuda12 >/dev/null 2>&1   # nvhpc only (conflicts with gcc module)
export CUDA_LIBDIR=$NVHPC_ROOT/cuda/lib64

NX_LIST="${NX_LIST:-64 128 256}"
BACKENDS="${BACKENDS:-pascal filtered filtered_v2}"
RHO=0.25
TMP=$(mktemp -d)

echo "host=$(hostname)  date=$(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1
echo "ghost-cell GPU order sweep — expect ratio->4, order->2, [rho]=0.25"

for backend in ${BACKENDS}; do
    echo ""
    printf "  %-12s %5s %5s %18s %8s %8s %8s\n" backend nx Nt L2_error ratio order rho
    PREV=""
    for nx in ${NX_LIST}; do
        INP="${TMP}/in_${backend}_${nx}.txt"
        LOG="${TMP}/log_${backend}_${nx}.txt"
        cat > "${INP}" <<EOF
nx = ${nx}
ny = ${nx}
nz = ${nx}
npx = 1
npy = 1
npz = 1
rho = ${RHO}
eps = 0.005
Tmax = 0.003
dt   = 0.001
option = order
tdma_backend = ${backend}
EOF
        mpirun -np 1 "${BIN}" "${INP}" >"${LOG}" 2>&1
        RC=$?
        L2=$(grep -Eo "Global L2 error = [0-9.eE+-]+" "${LOG}" | awk '{print $NF}')
        NT=$(grep -Eo "max_iter\] = [0-9]+" "${LOG}" | awk '{print $NF}')
        RHOP=$(grep -Eo "\[rho\] = [0-9.eE+-]+" "${LOG}" | awk '{print $NF}')
        L2="${L2:-FAIL}"; NT="${NT:-?}"; RHOP="${RHOP:-?}"
        RATIO="-"; ORDER="-"
        if [ "${L2}" != "FAIL" ] && [ -n "${PREV}" ]; then
            RATIO=$(awk -v a="${PREV}" -v b="${L2}" 'BEGIN{if(b>0)printf "%.3f",a/b; else print "nan"}')
            ORDER=$(awk -v r="${RATIO}" 'BEGIN{if(r>0)printf "%.3f",log(r)/log(2); else print "nan"}')
        fi
        printf "  %-12s %5s %5s %18s %8s %8s %8s (rc=%d)\n" \
               "${backend}" "${nx}" "${NT}" "${L2}" "${RATIO}" "${ORDER}" "${RHOP}" "${RC}"
        [ "${L2}" != "FAIL" ] && PREV="${L2}"
        [ "${L2}" == "FAIL" ] && { echo "    --- tail log ---"; tail -12 "${LOG}" | sed 's/^/    /'; }
    done
done
rm -rf "${TMP}"
echo ""
echo "[done]"
