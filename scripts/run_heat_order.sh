#!/bin/bash
#SBATCH -J HEAT_ORDER
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --ntasks-per-node=32
#SBATCH -o log/%x_%j.out
#SBATCH -e log/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --comment etc

# ============================================================
#  Heat equation — order-of-accuracy sweep
#
#  Loops over a set of grid sizes (NX) and a set of backends
#  (filtered / pascal), runs each case, scrapes the L2 error
#  from stdout, and writes a table + computes consecutive
#  ratios so a refinement factor of 2 in mesh gives the
#  observed convergence rate.
#
#  Usage (submit):
#    cd /scratch/x3319a05/Filtered_TDMA
#    mkdir -p log results
#    sbatch run_heat_order.sh
#
#  Or run interactively:
#    bash run_heat_order.sh
#
#  Override defaults via env vars:
#    NX_LIST="64 128 256 512"          # grid sizes
#    BACKENDS="filtered pascal"        # which backends to test
#    NPX=2 NPY=2 NPZ=2                  # MPI decomposition (NP=NPX*NPY*NPZ)
#    EPS=0.005                           # filter cutoff (filtered only)
#    RHO=0.25                            # rho -> dt = rho/(1-2 rho)*2 dx^2
# ============================================================

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
fi
cd "${SCRIPT_DIR}"

# --- Knobs -------------------------------------------------------
NX_LIST="${NX_LIST:-32 64 128 256 512}"
BACKENDS="${BACKENDS:-filtered pascal}"
NPX="${NPX:-2}"; NPY="${NPY:-2}"; NPZ="${NPZ:-2}"
NP=$(( NPX * NPY * NPZ ))
EPS="${EPS:-0.005}"
RHO="${RHO:-0.25}"

INPUT_DIR="${SCRIPT_DIR}/Heat/inputs/order_run"
RESULT_DIR="${SCRIPT_DIR}/results/heat_order"
RESULT_FILE="${RESULT_DIR}/order_$(date +%Y%m%d_%H%M%S).txt"

mkdir -p "${INPUT_DIR}" "${RESULT_DIR}" "${SCRIPT_DIR}/log"

# --- Modules + build --------------------------------------------
module purge
module load gcc/15.2.0
module load mpi/openmpi-4.1.8
module load fftw3/3.3.10

echo "[build] make heat"
make heat >/dev/null
BIN="${SCRIPT_DIR}/build/bin/heat.out"
[ -x "${BIN}" ] || { echo "[ERROR] missing binary ${BIN}" >&2; exit 1; }

# --- Header ------------------------------------------------------
{
echo "============================================================"
echo " Heat order-of-accuracy sweep"
echo " host=$(hostname)  date=$(date '+%F %T')"
echo " NP=${NP}  (NPX=${NPX} NPY=${NPY} NPZ=${NPZ})"
echo " RHO=${RHO}  EPS=${EPS}"
echo " NX_LIST=${NX_LIST}"
echo " BACKENDS=${BACKENDS}"
echo "============================================================"
} | tee "${RESULT_FILE}"

# --- Generate inputs --------------------------------------------
make_input() {
    local nx="$1" backend="$2" path="$3"
    cat > "${path}" <<EOF
!==================== meshes ====================
nx = ${nx}
ny = ${nx}
nz = ${nx}

!==================== procs =====================
npx = ${NPX}
npy = ${NPY}
npz = ${NPZ}

!==================== physics ===================
rho = ${RHO}
eps = ${EPS}

!==================== time ======================
Tmax = 0.003
dt   = 0.001

!==================== option ====================
option = order

!==================== TDMA backend ====================
tdma_backend = ${backend}
EOF
}

# --- Run sweep ---------------------------------------------------
declare -A ERR
for backend in ${BACKENDS}; do
    PREV_ERR=""
    PREV_NX=""
    {
    echo
    printf "  %-9s  %5s  %5s  %14s  %8s  %s\n" "backend" "nx" "Nt" "L2_error" "ratio" "order"
    printf "  %-9s  %5s  %5s  %14s  %8s  %s\n" "-------" "----" "---" "--------------" "--------" "-----"
    } | tee -a "${RESULT_FILE}"

    for nx in ${NX_LIST}; do
        INP="${INPUT_DIR}/PARA_INPUT_${backend}_${nx}.txt"
        make_input "${nx}" "${backend}" "${INP}"

        LOG="${RESULT_DIR}/log_${backend}_${nx}.txt"
        T0=$(date +%s)
        mpirun --bind-to none -np ${NP} "${BIN}" "${INP}" >"${LOG}" 2>&1
        RC=$?
        T1=$(date +%s)

        L2=$(grep -Eo "L2 error = [0-9.+\-eE]+" "${LOG}" | awk '{print $NF}')
        NT=$(grep -Eo "max_iter\] = [0-9]+"      "${LOG}" | awk '{print $NF}')
        L2="${L2:-FAIL}"
        NT="${NT:-?}"

        RATIO="-"
        ORDER="-"
        if [ "${L2}" != "FAIL" ] && [ -n "${PREV_ERR}" ]; then
            # ratio = E(h)/E(h/2);  order = log2(ratio)
            RATIO=$(awk -v a="${PREV_ERR}" -v b="${L2}" 'BEGIN{ if (b>0) printf "%.3f", a/b; else print "nan" }')
            ORDER=$(awk -v r="${RATIO}"   'BEGIN{ if (r>0) printf "%+.3f", log(r)/log(2); else print "nan" }')
        fi

        printf "  %-9s  %5s  %5s  %14s  %8s  %s  (rc=%d, %ds)\n" \
               "${backend}" "${nx}" "${NT}" "${L2}" "${RATIO}" "${ORDER}" "${RC}" "$((T1-T0))" \
               | tee -a "${RESULT_FILE}"

        ERR[${backend},${nx}]="${L2}"
        if [ "${L2}" != "FAIL" ]; then
            PREV_ERR="${L2}"
            PREV_NX="${nx}"
        fi
    done
done

{
echo
echo "[done] full log: ${RESULT_FILE}"
echo "       per-run logs: ${RESULT_DIR}/log_<backend>_<nx>.txt"
} | tee -a "${RESULT_FILE}"
