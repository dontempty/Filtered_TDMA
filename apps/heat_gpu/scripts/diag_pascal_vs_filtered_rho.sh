#!/bin/bash
source /usr/share/lmod/lmod/init/bash 2>/dev/null || source /etc/profile.d/lmod.sh 2>/dev/null
module purge 2>/dev/null
module load gcc/11.5.0 mpi/openmpi-4.1.8 nvhpc/25.11_cuda12 2>/dev/null
BIN=/scratch/x3319a05/Filtered_TDMA/build_sm90/bin/heat_gpu.out
NODE=$(hostname)
export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n
MPI_FLAGS="--mca pml ucx --mca osc ucx --mca coll ^hcoll --oversubscribe"
run_one() {
    local backend=$1 rho=$2 N=$3
    local inp=/scratch/x3319a05/Filtered_TDMA/apps/heat_gpu/scripts/_in_${backend}_${rho}_${N}.txt
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
    local out l2
    out=$(mpirun ${MPI_FLAGS} --host ${NODE}:8 -np 8 "${BIN}" "$inp" 2>/dev/null)
    l2=$(echo "$out" | grep -oP '(?<=Global L2 error = )\S+')
    printf "  %-12s rho=%s  N=%-4s  L2=%s\n" "${backend}" "${rho}" "${N}" "${l2:-FAIL}"
}
echo "# node=${NODE}"
for backend in pascal filtered_v1; do
    echo "=== backend=${backend} ==="
    for rho in 0.25 0.40; do
        for N in 256 512; do run_one "$backend" "$rho" "$N"; done
    done
done
echo "Done."
