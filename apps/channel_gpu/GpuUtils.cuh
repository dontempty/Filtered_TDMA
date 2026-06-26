#ifndef CHANNEL_GPU_UTILS_CUH
#define CHANNEL_GPU_UTILS_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace channel {

inline void cuda_check(cudaError_t e, const char* expr, const char* file, int line)
{
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[CUDA] %s failed at %s:%d: %s\n",
                     expr, file, line, cudaGetErrorString(e));
        std::fflush(stderr);
        std::abort();
    }
}

} // namespace channel

#define CHANNEL_CUDA_CHECK(expr) ::channel::cuda_check((expr), #expr, __FILE__, __LINE__)

#endif // CHANNEL_GPU_UTILS_CUH
