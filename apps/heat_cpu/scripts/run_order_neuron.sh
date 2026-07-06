#!/bin/bash
#SBATCH -J ftdma_order
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --comment etc
#SBATCH -o /scratch/x3319a05/Filtered_TDMA/apps/heat_cpu/results/order_accuracy/order_%j.out
#SBATCH -e /scratch/x3319a05/Filtered_TDMA/apps/heat_cpu/results/order_accuracy/order_%j.err
# =============================================================================
#  Order-of-accuracy verification for heat_cpu on Neuron CPU partition.
#
#  Varies grid size N=32,64,128,256 (np=1) for each backend x rho.
#  Prints L2 error per size; convergence slope should be ~2.0 (2nd order).
#
#  Submit:
#    sbatch apps/heat_cpu/scripts/run_order_neuron.sh
# =============================================================================

PROJ=/scratch/x3319a05/Filtered_TDMA
BIN="${PROJ}/build/bin/heat.out"
INP="${PROJ}/apps/heat_cpu/inputs/order_accuracy"

echo "============================================================"
echo " host : $(hostname)"
echo " date : $(date '+%F %T')"
echo " bin  : ${BIN}"
echo "============================================================"

module purge
module load gcc/15.2.0 mpi/openmpi-4.1.8

[ -x "${BIN}" ] || { echo "ERROR: missing ${BIN}"; exit 1; }

for rho_str in rho025 rho040; do
    echo ""
    echo "################## ${rho_str} ##################"
    for backend in pascal filtered filtered_v2; do
        echo "--- ${backend} ---"
        prev_err=""
        for N in 32 64 128 256; do
            inp="${INP}/in_${backend}_${rho_str}_n${N}.txt"
            [ -f "${inp}" ] || { echo "  N=${N}: [skip missing]"; continue; }
            result=$(mpirun -np 1 "${BIN}" "${inp}" 2>/dev/null | grep "Global L2 error")
            err=$(echo "${result}" | awk '{print $NF}')
            if [ -n "${prev_err}" ] && [ -n "${err}" ]; then
                order=$(awk "BEGIN {printf \"%.2f\", log(${prev_err}/${err})/log(2)}")
                echo "  N=${N}: ${err}   (order=${order})"
            else
                echo "  N=${N}: ${err}"
            fi
            prev_err="${err}"
        done
    done
done

echo ""
echo "==== DONE $(date '+%F %T') ===="
