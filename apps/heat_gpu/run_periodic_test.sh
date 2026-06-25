#!/bin/bash
# GPU periodic order-of-accuracy test (all-periodic). Run on an allocated node:
#   srun --jobid=<JID> --gres=gpu:2 -n1 bash apps/heat_gpu/run_periodic_test.sh
set -u
PROJ=/scratch/x3319a05/Filtered_TDMA
BIN="${PROJ}/build/bin/heat_gpu.out"
source /etc/profile.d/lmod.sh 2>/dev/null || true
# Runtime: nvhpc ONLY — the binary links nvhpc's bundled hpcx OpenMPI (CUDA-aware).
# Loading mpi/openmpi-4.1.8 too would shadow mpirun and break multi-GPU MPI.
module purge; module load nvhpc/25.11_cuda12 >/dev/null 2>&1
export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n
MPI_FLAGS="--mca pml ucx --mca osc ucx --mca coll ^hcoll --oversubscribe"
NODE=$(hostname)
TD=$(mktemp -d)

gen(){ # N npx npy npz backend periodic
cat > "$TD/in.txt" <<EOF
nx = $1
ny = $1
nz = $1
npx = $2
npy = $3
npz = $4
rho = 0.25
eps = 0.005
option = order
tdma_backend = $5
periodic = $6
EOF
}
run(){ # NP
  mpirun ${MPI_FLAGS} --host "${NODE}:${1}" -np $1 "${BIN}" "$TD/in.txt" 2>/dev/null \
    | grep -Eo "Global L2 error = [0-9.eE+-]+" | awk '{print $NF}'
}
sweep(){ # label npx npy npz NP periodic
  local lbl=$1 px=$2 py=$3 pz=$4 NP=$5 per=$6
  echo "===== ${lbl} (np=${NP}, periodic=${per}) ====="
  for be in pascal filtered filtered_v2; do
    printf "  %-12s" "$be"; prev=""
    for N in 64 128 256; do
      gen $N $px $py $pz $be $per
      L2=$(run $NP)
      if [ -n "$prev" ] && [ -n "$L2" ]; then o=$(awk -v a=$prev -v b=$L2 'BEGIN{printf "%.3f",log(a/b)/log(2)}'); else o="-"; fi
      printf "  N%s=%s(o=%s)" "$N" "${L2:-FAIL}" "$o"; prev=$L2
    done; echo
  done
}
echo "host=$NODE  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
sweep "PERIODIC 1GPU"      1 1 1 1 1
sweep "PERIODIC 2GPU zdec" 1 1 2 2 1
echo "===== DIRICHLET 1GPU regression (periodic=0) ====="
for be in pascal filtered_v2; do gen 64 1 1 1 $be 0
  printf "  %-12s N64=%s\n" "$be" "$(run 1)"; done
rm -rf "$TD"
