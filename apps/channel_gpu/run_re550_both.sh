#!/bin/bash
# Run channel_gpu Re550 — filtered (GPU 0,1) and pascal (GPU 2,3) simultaneously.
# Launch from login node (with nvhpc already loaded):
#   source env_gpu.sh
#   srun --jobid=<JID> --gres=gpu:4 -n1 --overlap bash apps/channel_gpu/run_re550_both.sh
set -u

PROJ=/scratch/x3319a05/Filtered_TDMA
BD="${PROJ}/build_sm90"
BIN="${BD}/bin/channel_gpu.out"
WDIR="${PROJ}/apps/channel_gpu"
NODE=$(hostname)

echo "=== channel_gpu Re550 on ${NODE}  [$(date '+%F %T')] ==="
echo "  nvcc:   $(which nvcc 2>&1)"
echo "  mpicxx: $(which mpicxx 2>&1)"
echo "  mpirun: $(which mpirun 2>&1)"
[ -x "${BIN}" ] || { echo "ERROR: ${BIN} missing"; exit 1; }

# ---- runtime env ----
unset PMIX_INSTALL_PREFIX
export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n
MPI_FLAGS="--bind-to none --mca pml ucx --mca osc ucx --mca coll ^hcoll --oversubscribe"

cd "${WDIR}"
mkdir -p log statistics/re550_f statistics/re550_p \
         instant/re550_f instant/re550_p \
         restart_out/re550_f restart_out/re550_p \
         restart_in/re550_f restart_in/re550_p

echo ""
echo "=== [1] Filtered  (GPU 0,1, np=2) ==="
t0=$(date +%s)
CUDA_VISIBLE_DEVICES=0,1 \
  mpirun ${MPI_FLAGS} --host "${NODE}:2" -np 2 \
  "${BIN}" input/PARA_INPUT_re550_filtered.dat \
  > log/re550_filtered.log 2>&1 &
PID_F=$!

echo "=== [2] Pascal    (GPU 2,3, np=2) ==="
CUDA_VISIBLE_DEVICES=2,3 \
  mpirun ${MPI_FLAGS} --host "${NODE}:2" -np 2 \
  "${BIN}" input/PARA_INPUT_re550_pascal.dat \
  > log/re550_pascal.log 2>&1 &
PID_P=$!

echo "  filtered PID=${PID_F}   pascal PID=${PID_P}"
echo "  logs: log/re550_filtered.log  log/re550_pascal.log"
echo "  waiting for both..."

wait ${PID_F}; RC_F=$?
t1=$(date +%s)
echo "[filtered done]  rc=${RC_F}  wall=$((t1-t0))s"

wait ${PID_P}; RC_P=$?
t2=$(date +%s)
echo "[pascal   done]  rc=${RC_P}  wall=$((t2-t0))s"

echo ""
echo "=== tail of filtered log ==="
tail -20 log/re550_filtered.log
echo ""
echo "=== tail of pascal log ==="
tail -20 log/re550_pascal.log
echo ""
echo "==== ALL DONE [$(date '+%F %T')] ===="
