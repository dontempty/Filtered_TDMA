#!/bin/bash
# Run channel_gpu Re180 — pascal then filtered, SEQUENTIAL, each np=2 on GPU 0,1.
# (Only 2 GPUs available, and the re180 inputs use np3=2, so we can't run both at once.)
# Launch from login node (with nvhpc already loaded):
#   source env_gpu.sh
#   srun --jobid=<JID> --overlap -n1 -c 16 bash apps/channel_gpu/run_re180_both.sh
set -u

PROJ=/scratch/x3319a05/Filtered_TDMA
BD="${PROJ}/build_sm90"
BIN="${BD}/bin/channel_gpu.out"
WDIR="${PROJ}/apps/channel_gpu"
NODE=$(hostname)

echo "=== channel_gpu Re180 on ${NODE}  [$(date '+%F %T')] ==="
echo "  nvcc:   $(which nvcc 2>&1)"
echo "  mpirun: $(which mpirun 2>&1)"
[ -x "${BIN}" ] || { echo "ERROR: ${BIN} missing"; exit 1; }

# ---- runtime env ----
unset PMIX_INSTALL_PREFIX
export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n
MPI_FLAGS="--bind-to none --mca pml ucx --mca osc ucx --mca coll ^hcoll --oversubscribe"

cd "${WDIR}"
mkdir -p log statistics/re180_f statistics/re180_p \
         instant/re180_f instant/re180_p \
         restart_out/re180_f restart_out/re180_p \
         restart_in/re180_f restart_in/re180_p

echo ""
echo "=== [1] Pascal   (GPU 0,1, np=2) ==="
t0=$(date +%s)
CUDA_VISIBLE_DEVICES=0,1 \
  mpirun ${MPI_FLAGS} --host "${NODE}:2" -np 2 \
  "${BIN}" input/PARA_INPUT_re180_pascal.dat \
  > log/re180_pascal.log 2>&1
RC_P=$?
t1=$(date +%s)
echo "[pascal   done]  rc=${RC_P}  wall=$((t1-t0))s"

echo ""
echo "=== [2] Filtered (GPU 0,1, np=2) ==="
CUDA_VISIBLE_DEVICES=0,1 \
  mpirun ${MPI_FLAGS} --host "${NODE}:2" -np 2 \
  "${BIN}" input/PARA_INPUT_re180_filtered.dat \
  > log/re180_filtered.log 2>&1
RC_F=$?
t2=$(date +%s)
echo "[filtered done]  rc=${RC_F}  wall=$((t2-t1))s"

echo ""
echo "=== tail of pascal log ==="
tail -8 log/re180_pascal.log
echo ""
echo "=== tail of filtered log ==="
tail -8 log/re180_filtered.log
echo ""
echo "==== ALL DONE [$(date '+%F %T')] ===="
