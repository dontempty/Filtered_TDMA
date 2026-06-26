# Source this to set up the GPU build/run toolchain on KISTI Neuron.
# Avoids the nvhpc<->gcc module conflict by loading nvhpc only.
source /etc/profile.d/lmod.sh 2>/dev/null || true
module purge 2>/dev/null
module load nvhpc/25.11_cuda12 2>/dev/null
NVR=/apps/compiler/nvidia_hpc_sdk/25.11/Linux_x86_64/25.11
export NVHPC_ROOT=$NVR
export PATH=$NVR/cuda/12.9/bin:$NVR/comm_libs/12.9/hpcx/hpcx-2.25.1/ompi/bin:$NVR/compilers/bin:$PATH
export LD_LIBRARY_PATH=$NVR/cuda/12.9/lib64:$NVR/comm_libs/12.9/hpcx/hpcx-2.25.1/ompi/lib:$NVR/math_libs/12.9/lib64:$NVR/compilers/lib:$LD_LIBRARY_PATH
