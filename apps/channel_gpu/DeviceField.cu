#include "DeviceField.hpp"
namespace channel {

namespace gpu_kernels {

__global__ void k_fill(double* p, std::size_t n, double v)
{
    std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t stride = blockDim.x * gridDim.x;
    for (std::size_t q = tid; q < n; q += stride) p[q] = v;
}

} // namespace gpu_kernels
using namespace gpu_kernels;

void DeviceField::fill(double v)
{
    if (!ptr_ || n_ == 0) return;
    int block = 256;
    int grid = static_cast<int>((n_ + block - 1) / block);
    if (grid > 65535) grid = 65535;
    k_fill<<<grid, block>>>(ptr_, n_, v);
    CHANNEL_CUDA_CHECK(cudaGetLastError());
}

} // namespace channel
