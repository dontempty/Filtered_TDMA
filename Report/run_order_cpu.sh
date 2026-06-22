#!/bin/bash
# Order-of-accuracy sweep for the ghost-cell heat_cpu solver.
# Usage: bash run_order_cpu.sh "NPX NPY NPZ" "backend1 backend2" "nx1 nx2 ..."
set -u
cd /scratch/x3319a05/Filtered_TDMA
module purge 2>/dev/null
module load gcc/15.2.0 mpi/openmpi-4.1.8 2>/dev/null

read NPX NPY NPZ <<< "${1:-1 1 1}"
BACKENDS="${2:-pascal filtered}"
NX_LIST="${3:-64 128 256 512}"
NP=$(( NPX * NPY * NPZ ))
RHO="${RHO:-0.25}"

TMP=$(mktemp -d)
BIN=build/bin/heat.out

for backend in ${BACKENDS}; do
    printf "\n  %-10s np=%d (%d,%d,%d)  rho=%s\n" "$backend" "$NP" "$NPX" "$NPY" "$NPZ" "$RHO"
    printf "  %5s %6s %16s %8s %8s %10s\n" "nx" "Nt" "L2_error" "ratio" "order" "rho_print"
    PREV=""
    for nx in ${NX_LIST}; do
        INP="$TMP/in_${backend}_${nx}.txt"
        cat > "$INP" <<EOF
nx = ${nx}
ny = ${nx}
nz = ${nx}
npx = ${NPX}
npy = ${NPY}
npz = ${NPZ}
rho = ${RHO}
eps = 0.005
Tmax = 0.003
dt   = 0.001
option = order
tdma_backend = ${backend}
EOF
        LOG="$TMP/log_${backend}_${nx}.txt"
        mpirun --bind-to none -np ${NP} "$BIN" "$INP" > "$LOG" 2>&1
        L2=$(grep -Eo "Global L2 error = [0-9.eE+\-]+" "$LOG" | awk '{print $NF}')
        NT=$(grep -Eo "max_iter\] = [0-9]+" "$LOG" | awk '{print $NF}')
        RHOP=$(grep -Eo "\[rho\] = [0-9.eE+\-]+" "$LOG" | awk '{print $NF}')
        L2="${L2:-FAIL}"; NT="${NT:-?}"; RHOP="${RHOP:-?}"
        RATIO="-"; ORDER="-"
        if [ "$L2" != "FAIL" ] && [ -n "$PREV" ]; then
            RATIO=$(awk -v a="$PREV" -v b="$L2" 'BEGIN{if(b>0)printf "%.3f",a/b; else print "nan"}')
            ORDER=$(awk -v r="$RATIO" 'BEGIN{if(r>0)printf "%+.3f",log(r)/log(2); else print "nan"}')
        fi
        printf "  %5s %6s %16s %8s %8s %10s\n" "$nx" "$NT" "$L2" "$RATIO" "$ORDER" "$RHOP"
        [ "$L2" != "FAIL" ] && PREV="$L2"
        [ "$L2" == "FAIL" ] && { echo "    --- last 15 lines of log ---"; tail -15 "$LOG" | sed 's/^/    /'; }
    done
done
rm -rf "$TMP"
