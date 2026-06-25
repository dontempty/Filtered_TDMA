#!/bin/bash
source /usr/share/lmod/lmod/init/bash
module load gcc/11.5.0 mpi/openmpi-4.1.8 nvhpc/25.11_cuda12
cd /scratch/x3319a05/Filtered_TDMA

nvcc -O3 -std=c++17 -Xcompiler -fPIC -arch=sm_90 -ccbin mpicxx \
  -Ibuild_sm90/include \
  libs/filtered_tdma/filtered_tdma_cuda.cu \
  -c -o build_sm90/obj/ftdma/filtered_tdma_cuda.o 2>&1 | grep -v "^$"

ar rcs build_sm90/lib/libfiltered_tdma.a \
  build_sm90/obj/ftdma/filtered_tdma_cuda.o \
  build_sm90/obj/ftdma/filtered_tdma_cycl.o \
  build_sm90/obj/ftdma/filtered_tdma.o \
  build_sm90/obj/ftdma/filtered_tdma_profile.o

CUDA_LIB=/apps/compiler/nvidia_hpc_sdk/25.11/Linux_x86_64/25.11/cuda/lib64
CUDA_INC=/apps/compiler/nvidia_hpc_sdk/25.11/Linux_x86_64/25.11/cuda/include
mpicxx -O3 -std=c++17 -fPIC -march=x86-64-v3 \
  -Iapps/heat_gpu -Ibuild_sm90/include -I${CUDA_INC} \
  build_sm90/obj/exgpu_global.o build_sm90/obj/exgpu_main.o \
  build_sm90/obj/exgpu_mpi_subdomain.o build_sm90/obj/exgpu_mpi_topology.o \
  build_sm90/obj/exgpu_ghostcell_cuda.o build_sm90/obj/exgpu_solve_theta.o \
  build_sm90/obj/exgpu_tdma_backend_gpu.o \
  -o build_sm90/bin/heat_gpu.out \
  -Lbuild_sm90/lib -lfiltered_tdma -lpascal_tdma \
  -L${CUDA_LIB} -lcudart -lstdc++ 2>&1 | grep -v "^$"
echo "Build: $(ls -lh build_sm90/bin/heat_gpu.out | awk '{print $5, $9}')"

export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n
NODE=$(hostname)
MPI="mpirun --mca pml ucx --mca osc ucx --mca coll ^hcoll --oversubscribe --host ${NODE}:8"

echo "=== np=2 filtered_v2 256x256x2048 ==="
$MPI -np 2 build_sm90/bin/heat_gpu.out \
  apps/heat_gpu/inputs/scaling/strong_256x256x2048_np2_filtered_v2_rho025.txt \
  2>&1 | grep -E "DEBUG|CUDA\]|backend\]|Tmax\]"

echo "=== np=4 filtered_v2 256x256x2048 ==="
$MPI -np 4 build_sm90/bin/heat_gpu.out \
  apps/heat_gpu/inputs/scaling/strong_256x256x2048_np4_filtered_v2_rho025.txt \
  2>&1 | grep -E "DEBUG|CUDA\]|backend\]|Tmax\]"
