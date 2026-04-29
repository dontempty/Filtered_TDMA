#!/bin/sh
#SBATCH -J KISTI_TDMA
#SBATCH -p cpu
#SBATCH -N 1 # number of node
#SBATCH -n 32 # total process
#SBATCH --ntasks-per-node=32
#SBATCH -o log/%x_%j.out
#SBATCH -e log/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --comment etc

# ============================================================
#  Batch script for Filtered_TDMA channel solver
#
#  Usage (submit):
#    cd /scratch/x3319a05/Filtered_TDMA
#    mkdir -p log          # must exist before sbatch
#    sbatch channel.sh [PARA_INPUT.dat]
#
#  Or run interactively (no SLURM):
#    bash channel.sh [PARA_INPUT.dat]
#
#  NP = np1 * np2 * np3 is read automatically from PARA_INPUT.dat.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Resolve the project root -----------------------------------
# When run via sbatch, SLURM copies the script to a spool directory,
# so BASH_SOURCE[0] gives the wrong path.
# SLURM_SUBMIT_DIR is always set to the directory where sbatch was called.
if [ -n "${SLURM_SUBMIT_DIR}" ]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
fi

# --- Input file -------------------------------------------------
INPUT="${1:-${SCRIPT_DIR}/channel/PARA_INPUT.dat}"
if [ ! -f "${INPUT}" ]; then
    echo "[ERROR] Input file not found: ${INPUT}" >&2
    exit 1
fi

# --- Load modules -----------------------------------------------
module purge
module load gcc/15.2.0
module load mpi/openmpi-4.1.8
module load fftw3/3.3.10

make clean
make all

# --- Parse NP from PARA_INPUT.dat (np1 * np2 * np3) -------------
_get_val() {
    # $1: key name
    grep -E "^\s*${1}\s*=" "${INPUT}" \
        | sed 's/[^=]*=//; s/#.*//' \
        | tr -d '[:space:]' \
        | head -1
}

NP1=$(_get_val "np1"); NP1=${NP1:-1}
NP2=$(_get_val "np2"); NP2=${NP2:-1}
NP3=$(_get_val "np3"); NP3=${NP3:-1}
NP=$(( NP1 * NP2 * NP3 ))

echo "=============================================="
echo " Job     : ${SLURM_JOB_NAME:-interactive}  (ID=${SLURM_JOB_ID:-none})"
echo " Input   : ${INPUT}"
echo " MPI     : np1=${NP1}  np2=${NP2}  np3=${NP3}  -> NP=${NP}"
echo " Binary  : ${SCRIPT_DIR}/build/bin/channel.out"
echo "=============================================="

# --- Create output directories (solver also does this, but do it
#     here too so the paths exist before the first MPI launch) ----
_resolve_dir() {
    local key="$1" default="$2"
    local val
    val=$(_get_val "${key}")
    val="${val:-${default}}"
    # Relative paths are relative to channel/ (the run directory)
    if [[ "${val}" != /* ]]; then
        val="${SCRIPT_DIR}/channel/${val}"
    fi
    mkdir -p "${val}"
}

_resolve_dir "dir_cont_filein"  "restart_in"
_resolve_dir "dir_cont_fileout" "restart_out"
_resolve_dir "dir_instantfield" "instant"
_resolve_dir "dir_statistics"   "statistics"
mkdir -p "${SCRIPT_DIR}/log"

# --- Run --------------------------------------------------------
cd "${SCRIPT_DIR}/channel" || exit 1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting  mpirun -np ${NP}"
mpirun --bind-to none -np ${NP} "${SCRIPT_DIR}/build/bin/channel.out" "${INPUT}"
EXIT_CODE=$?
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished  (exit=${EXIT_CODE})"

exit ${EXIT_CODE}
