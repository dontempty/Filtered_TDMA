#!/bin/bash
# Order-of-accuracy sweep for heat_gpu on V100 (1 GPU, np=1).
#   option=order: dt = rho/(1-2rho)*2dx^2, Tmax fixed -> Nt = 2,8,32,128 for
#   nx = 64,128,256,512.  exact = sin sin sin exp(-3 pi^2 t) + cos cos cos.
#   Refining nx by 2 should drop the L2 error by ~4  => observed order ~2.
# Run on an allocated GPU node, e.g.:
#   srun --jobid=<JOBID> --gres=gpu:1 -n1 bash apps/heat_gpu/run_order_sweep.sh
set -u

PROJ=/scratch/x3319a05/Filtered_TDMA
BIN="${PROJ}/build/bin/heat_gpu.out"
OUTDIR="${PROJ}/apps/heat_gpu/results/order_v100"
mkdir -p "${OUTDIR}"

source /etc/profile.d/lmod.sh 2>/dev/null || true
module purge
module load gcc/15.2.0 mpi/openmpi-4.1.8 nvhpc/25.11_cuda12 >/dev/null 2>&1

NX_LIST="${NX_LIST:-64 128 256 512}"
BACKENDS="${BACKENDS:-pascal filtered filtered_v2}"
RHO="${RHO:-0.25}"
EPS="${EPS:-0.005}"

gen_input() {
    local nx="$1" backend="$2" path="$3"
    cat > "${path}" <<EOF
nx = ${nx}
ny = ${nx}
nz = ${nx}
npx = 1
npy = 1
npz = 1
rho = ${RHO}
eps = ${EPS}
Tmax = 0.003
dt   = 0.001
option = order
tdma_backend = ${backend}
EOF
}

echo "host=$(hostname)  date=$(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1
echo "RHO=${RHO}  EPS=${EPS}  NX_LIST=${NX_LIST}"
echo "(order = log2( E(h) / E(h/2) ); central diff => expect ~2)"

for backend in ${BACKENDS}; do
    echo ""
    printf "  %-12s %5s %5s %20s %8s %8s\n" backend nx Nt L2_error ratio order
    printf "  %-12s %5s %5s %20s %8s %8s\n" "------------" "-----" "-----" "--------------------" "--------" "--------"
    PREV=""
    for nx in ${NX_LIST}; do
        INP="${OUTDIR}/in_${backend}_${nx}.txt"
        LOG="${OUTDIR}/log_${backend}_${nx}.txt"
        gen_input "${nx}" "${backend}" "${INP}"
        T0=$(date +%s)
        mpirun -np 1 "${BIN}" "${INP}" >"${LOG}" 2>&1
        RC=$?
        T1=$(date +%s)
        L2=$(grep -Eo "Global L2 error = [0-9.eE+-]+" "${LOG}" | awk '{print $NF}')
        NT=$(grep -Eo "max_iter\] = [0-9]+" "${LOG}" | awk '{print $NF}')
        L2="${L2:-FAIL}"; NT="${NT:-?}"
        RATIO="-"; ORDER="-"
        if [ "${L2}" != "FAIL" ] && [ -n "${PREV}" ]; then
            RATIO=$(awk -v a="${PREV}" -v b="${L2}" 'BEGIN{if(b>0)printf "%.3f",a/b; else print "nan"}')
            ORDER=$(awk -v r="${RATIO}" 'BEGIN{if(r>0)printf "%.3f",log(r)/log(2); else print "nan"}')
        fi
        printf "  %-12s %5s %5s %20s %8s %8s  (rc=%d, %ds)\n" \
               "${backend}" "${nx}" "${NT}" "${L2}" "${RATIO}" "${ORDER}" "${RC}" "$((T1-T0))"
        [ "${L2}" != "FAIL" ] && PREV="${L2}"
    done
done
echo ""
echo "[done] per-run logs in ${OUTDIR}/log_<backend>_<nx>.txt"
