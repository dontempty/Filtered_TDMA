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
    local inp=/scratch/x3319a05/Filtered_TDMA/apps/heat_gpu/scripts/_vb_${backend}_${rho}_${N}.txt
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
    echo "${l2:-FAIL}"
}
echo "# node=${NODE}  (np=8, 2x2x2, fixed binary $(date '+%H:%M'))"
for backend in pascal filtered_v1 filtered_v2; do
  echo "=== backend=${backend} ==="
  for rho in 0.25 0.40; do
    prev=""
    printf "  rho=%s :\n" "$rho"
    for N in 64 128 256 512; do
      l2=$(run_one "$backend" "$rho" "$N")
      ratio="-"
      if [ -n "$prev" ] && [ "$l2" != "FAIL" ]; then
        ratio=$(awk -v a="$prev" -v b="$l2" 'BEGIN{if(b>0)printf "%.2f",a/b; else print "nan"}')
      fi
      printf "    N=%-4s  L2=%-14s  ratio=%s\n" "$N" "$l2" "$ratio"
      [ "$l2" != "FAIL" ] && prev="$l2"
    done
  done
done
echo "Done."
