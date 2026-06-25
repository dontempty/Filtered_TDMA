#!/bin/bash
# Order-of-accuracy test: filtered_v1 and filtered_v2, N=64~512, np=8, 2x2x2 decomp

if ! type module &>/dev/null; then
    source /usr/share/lmod/lmod/init/bash
    module load gcc/11.5.0 mpi/openmpi-4.1.8 nvhpc/25.11_cuda12
fi

PROJ=/scratch/x3319a05/Filtered_TDMA
BIN="${PROJ}/build_sm90/bin/heat_gpu.out"
RES="${PROJ}/apps/heat_gpu/results/order_accuracy"
mkdir -p "${RES}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="${RES}/order_np8_222_${TIMESTAMP}.txt"
NODE=$(hostname)

export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n
MPI_FLAGS="--mca pml ucx --mca osc ucx --mca coll ^hcoll --oversubscribe"

{
echo "# Order-of-accuracy  np=8  decomp=2x2x2  $(date)"
echo "# Binary: $(ls -lh ${BIN} | awk '{print $5, $6, $7, $8}')"
} | tee "${OUTFILE}"

run_order() {
    local backend=$1 rho=$2 N=$3
    local inp=/tmp/order_${backend}_rho${rho}_N${N}.txt
    cat > "$inp" << INP
nx = ${N}
ny = ${N}
nz = ${N}
npx = 2
npy = 2
npz = 2
rho = ${rho}
eps = 0.005
Nt = 1
warmup = 0
option = order
tdma_backend = ${backend}
INP
    local out
    out=$(mpirun ${MPI_FLAGS} --host ${NODE}:8 -np 8 "${BIN}" "$inp" 2>/dev/null)
    local l2
    l2=$(echo "$out" | grep -oP '(?<=Global L2 error = )\S+')
    echo "  N=${N}  L2=${l2}" | tee -a "${OUTFILE}"
}

for backend in filtered_v1 filtered_v2; do
    for rho in 0.25 0.40; do
        echo "" | tee -a "${OUTFILE}"
        echo "=== backend=${backend}  rho=${rho} ===" | tee -a "${OUTFILE}"
        for N in 64 128 256 512; do
            run_order "$backend" "$rho" "$N"
        done
    done
done

echo ""
echo "Done.  Results: ${OUTFILE}"
