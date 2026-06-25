#!/bin/bash
#SBATCH -J h200_cp2
#SBATCH -p amd_h200nv_8
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --comment=inhouse
#SBATCH -o /scratch/x3319a05/Filtered_TDMA/apps/heat_gpu/log/comm_pure_v2_%j.log
#SBATCH -e /scratch/x3319a05/Filtered_TDMA/apps/heat_gpu/log/comm_pure_v2_%j.err

source /usr/share/lmod/lmod/init/bash
module load gcc/11.5.0 mpi/openmpi-4.1.8 nvhpc/25.11_cuda12

PROJ=/scratch/x3319a05/Filtered_TDMA
cd "${PROJ}"
echo "=== [$(date '+%F %T')] Build on $(hostname) ==="

NVCC_FLAGS="-O3 -std=c++17 -Xcompiler -fPIC -arch=sm_90 -ccbin mpicxx"
INCDIR=build_sm90/include

# Step 1: copy updated headers first
cp libs/pascal_tdma/pascal_tdma_many_cuda.hpp  ${INCDIR}/
cp libs/filtered_tdma/filtered_tdma_cuda.hpp   ${INCDIR}/
cp apps/heat_gpu/tdma_backend_gpu.hpp          ${INCDIR}/
echo "Headers copied."

# Step 2: compile modified sources
nvcc ${NVCC_FLAGS} -I${INCDIR} \
    libs/pascal_tdma/pascal_tdma_many_cuda.cu \
    -c -o build_sm90/obj/pascal/pascal_tdma_many_cuda.o && echo "pascal OK" || { echo "pascal FAIL"; exit 1; }

nvcc ${NVCC_FLAGS} -I${INCDIR} \
    libs/filtered_tdma/filtered_tdma_cuda.cu \
    -c -o build_sm90/obj/ftdma/filtered_tdma_cuda.o && echo "filtered OK" || { echo "filtered FAIL"; exit 1; }

nvcc ${NVCC_FLAGS} -Iapps/heat_gpu -I${INCDIR} \
    apps/heat_gpu/solve_theta.cu \
    -c -o build_sm90/obj/exgpu_solve_theta.o && echo "solve_theta OK" || { echo "solve_theta FAIL"; exit 1; }

# Step 3: re-archive
POBJ=build_sm90/obj/pascal
FOBJ=build_sm90/obj/ftdma
ar rcs build_sm90/lib/libpascal_tdma.a \
    ${POBJ}/pascal_tdma_many_cuda.o ${POBJ}/pascal_tdma_many.o \
    ${POBJ}/pascal_tdma_single.o   ${POBJ}/para_range.o \
    ${POBJ}/tdma_local.o           ${POBJ}/tdma_local_cuda.o && echo "libpascal OK"

ar rcs build_sm90/lib/libfiltered_tdma.a \
    ${FOBJ}/filtered_tdma_cuda.o   ${FOBJ}/filtered_tdma_cycl.o \
    ${FOBJ}/filtered_tdma.o        ${FOBJ}/filtered_tdma_profile.o && echo "libfiltered OK"

# Step 4: re-link
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
strings build_sm90/bin/heat_gpu.out | grep "tdma_z_gpu"

# Step 5: sanity check
export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n
MPI_FLAGS="--mca pml ucx --mca osc ucx --mca coll ^hcoll --oversubscribe"
NODE=$(hostname)

export TIMING_CSV=/tmp/sanity_v2.csv
mpirun ${MPI_FLAGS} --host ${NODE}:8 -np 2 build_sm90/bin/heat_gpu.out \
    apps/heat_gpu/inputs/scaling/strong_128x128x1024_np2_filtered_v2_rho025.txt 2>&1 \
    | grep -E "backend\]|Tmax\]" || true
echo "--- z_comm | z_gpu (ms) ---"
grep -E "tdma_z_comm|tdma_z_gpu" /tmp/sanity_v2.csv | awk -F, 'NR<=6{print $3, $4}'

# -----------------------------------------------------------------------
BIN="${PROJ}/build_sm90/bin/heat_gpu.out"
INP="${PROJ}/apps/heat_gpu/inputs/scaling"
RES="${PROJ}/apps/heat_gpu/results/comm_pure/h200"
mkdir -p "${RES}"

run_case() {
    local np=$1 inp_file=$2
    local tag; tag=$(basename "${inp_file}" .txt)
    [ -f "${inp_file}" ] || { echo "  [skip] ${tag}"; return 0; }
    echo "---- np=${np}  [$(date +%H:%M:%S)]  ${tag} ----"
    TIMING_CSV="${RES}/timing_${tag}.csv" \
        mpirun ${MPI_FLAGS} --host ${NODE}:8 -np ${np} "${BIN}" "${inp_file}" 2>&1 \
        | grep -E "rho\]|backend\]|Tmax\]" || true
}

echo ""; echo "===== STRONG SCALING pascal ====="
for rho in rho025 rho040; do for np in 2 4 8; do
    run_case ${np} "${INP}/strong_64x64x512_np${np}_pascal_${rho}.txt"
    run_case ${np} "${INP}/strong_128x128x1024_np${np}_pascal_${rho}.txt"
    run_case ${np} "${INP}/strong_256x256x2048_np${np}_pascal_${rho}.txt"
done; done

echo ""; echo "===== STRONG SCALING filtered_v2 ====="
for rho in rho025 rho040; do for np in 2 4 8; do
    run_case ${np} "${INP}/strong_64x64x512_np${np}_filtered_v2_${rho}.txt"
    run_case ${np} "${INP}/strong_128x128x1024_np${np}_filtered_v2_${rho}.txt"
    run_case ${np} "${INP}/strong_256x256x2048_np${np}_filtered_v2_${rho}.txt"
done; done

echo ""; echo "===== WEAK SCALING pascal ====="
for rho in rho025 rho040; do for np in 2 4 8; do
    run_case ${np} "${INP}/weak_128cube_np${np}_pascal_${rho}.txt"
    run_case ${np} "${INP}/weak_256cube_np${np}_pascal_${rho}.txt"
    run_case ${np} "${INP}/weak_512cube_np${np}_pascal_${rho}.txt"
done; done

echo ""; echo "===== WEAK SCALING filtered_v2 ====="
for rho in rho025 rho040; do for np in 2 4 8; do
    run_case ${np} "${INP}/weak_128cube_np${np}_filtered_v2_${rho}.txt"
    run_case ${np} "${INP}/weak_256cube_np${np}_filtered_v2_${rho}.txt"
    run_case ${np} "${INP}/weak_512cube_np${np}_filtered_v2_${rho}.txt"
done; done

echo ""; echo "===== REFINE SCALING pascal np=2 ====="
for inp in $(ls "${INP}"/refine_*_np2_pascal_*.txt | sort); do
    run_case 2 "${inp}"
done

echo ""; echo "===== REFINE SCALING filtered_v2 np=2 ====="
for inp in $(ls "${INP}"/refine_*_np2_filtered_v2_*.txt | sort); do
    run_case 2 "${inp}"
done

echo ""
echo "==== ALL DONE [$(date '+%F %T')]  results: ${RES} ===="
echo "CSVs: $(ls ${RES}/timing_*.csv 2>/dev/null | wc -l)"
