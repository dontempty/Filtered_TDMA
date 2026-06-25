#!/bin/bash
#SBATCH -J h200_fix6
#SBATCH -p amd_h200nv_8
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=01:00:00
#SBATCH --comment=inhouse
#SBATCH -o /scratch/x3319a05/Filtered_TDMA/apps/heat_gpu/log/comm_pure_fix_%j.log
#SBATCH -e /scratch/x3319a05/Filtered_TDMA/apps/heat_gpu/log/comm_pure_fix_%j.err

source /usr/share/lmod/lmod/init/bash
module load gcc/11.5.0 mpi/openmpi-4.1.8 nvhpc/25.11_cuda12

PROJ=/scratch/x3319a05/Filtered_TDMA
cd "${PROJ}"
echo "=== [$(date '+%F %T')] Build on $(hostname) ==="

NVCC_FLAGS="-O3 -std=c++17 -Xcompiler -fPIC -arch=sm_90 -ccbin mpicxx"
INCDIR=build_sm90/include

cp libs/filtered_tdma/filtered_tdma_cuda.hpp  ${INCDIR}/

nvcc ${NVCC_FLAGS} -I${INCDIR} \
    libs/filtered_tdma/filtered_tdma_cuda.cu \
    -c -o build_sm90/obj/ftdma/filtered_tdma_cuda.o && echo "filtered OK" || { echo "filtered FAIL"; exit 1; }

ar rcs build_sm90/lib/libfiltered_tdma.a \
    build_sm90/obj/ftdma/filtered_tdma_cuda.o \
    build_sm90/obj/ftdma/filtered_tdma_cycl.o \
    build_sm90/obj/ftdma/filtered_tdma.o \
    build_sm90/obj/ftdma/filtered_tdma_profile.o && echo "libfiltered OK"

CUDA_INC=/apps/compiler/nvidia_hpc_sdk/25.11/Linux_x86_64/25.11/cuda/include
CUDA_LIB=/apps/compiler/nvidia_hpc_sdk/25.11/Linux_x86_64/25.11/cuda/lib64
BOBJ=build_sm90/obj
mpicxx -O3 -std=c++17 -fPIC -march=x86-64-v3 \
    -Iapps/heat_gpu -I${INCDIR} -I${CUDA_INC} \
    ${BOBJ}/exgpu_global.o ${BOBJ}/exgpu_main.o \
    ${BOBJ}/exgpu_mpi_subdomain.o ${BOBJ}/exgpu_mpi_topology.o \
    ${BOBJ}/exgpu_ghostcell_cuda.o ${BOBJ}/exgpu_solve_theta.o \
    ${BOBJ}/exgpu_tdma_backend_gpu.o \
    -o build_sm90/bin/heat_gpu.out \
    -Lbuild_sm90/lib -lfiltered_tdma -lpascal_tdma \
    -L${CUDA_LIB} -lcudart -lstdc++ && echo "Link OK" || { echo "Link FAIL"; exit 1; }

echo "=== Binary: $(ls -lh build_sm90/bin/heat_gpu.out) ==="

export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n
MPI_FLAGS="--mca pml ucx --mca osc ucx --mca coll ^hcoll --oversubscribe"
NODE=$(hostname)

BIN="${PROJ}/build_sm90/bin/heat_gpu.out"
INP="${PROJ}/apps/heat_gpu/inputs/scaling"
RES="${PROJ}/apps/heat_gpu/results/comm_pure/h200"

run_case() {
    local np=$1 inp_file=$2
    local tag; tag=$(basename "${inp_file}" .txt)
    [ -f "${inp_file}" ] || { echo "  [skip] ${tag}"; return 0; }
    echo "---- np=${np}  [$(date +%H:%M:%S)]  ${tag} ----"
    TIMING_CSV="${RES}/timing_${tag}.csv" \
        mpirun ${MPI_FLAGS} --host ${NODE}:8 -np ${np} "${BIN}" "${inp_file}" 2>&1 \
        | grep -E "rho\]|backend\]|Tmax\]" || true
}

echo ""; echo "===== 6 missing filtered_v2 cases ====="
for rho in rho025 rho040; do
    run_case 2 "${INP}/strong_256x256x2048_np2_filtered_v2_${rho}.txt"
    run_case 4 "${INP}/strong_256x256x2048_np4_filtered_v2_${rho}.txt"
    run_case 2 "${INP}/refine_256x256x1024_np2_filtered_v2_${rho}.txt"
done

echo ""
echo "==== DONE [$(date '+%F %T')] ===="
echo "CSVs in ${RES}: $(ls ${RES}/timing_*.csv 2>/dev/null | wc -l)"
