#ifndef NVTX_UTIL_HPP
#define NVTX_UTIL_HPP
//
// NVTX helper: USE_NVTX-gated push/pop and RAII scoped ranges.
// Compile with -DUSE_NVTX and link -lnvToolsExt to enable.
// Without USE_NVTX all macros vanish to no-ops; without the link
// nothing is referenced either.
//

#ifdef USE_NVTX
#  include <nvtx3/nvToolsExt.h>

inline void nvtx_push(const char* name) { nvtxRangePushA(name); }
inline void nvtx_pop()                  { nvtxRangePop(); }

// RAII scoped range so we don't have to remember to pop on every return path.
struct NVTXScopedRange {
    explicit NVTXScopedRange(const char* name) { nvtxRangePushA(name); }
    ~NVTXScopedRange()                         { nvtxRangePop(); }
    NVTXScopedRange(const NVTXScopedRange&)            = delete;
    NVTXScopedRange& operator=(const NVTXScopedRange&) = delete;
};

#  define NVTX_PUSH(name)   ::nvtx_push(name)
#  define NVTX_POP()        ::nvtx_pop()
#  define NVTX_CONCAT2(a,b) a##b
#  define NVTX_CONCAT(a,b)  NVTX_CONCAT2(a,b)
#  define NVTX_SCOPE(name)  NVTXScopedRange NVTX_CONCAT(_nvtx_range_, __LINE__)(name)
#else
#  define NVTX_PUSH(name)   ((void)0)
#  define NVTX_POP()        ((void)0)
#  define NVTX_SCOPE(name)  ((void)0)
#endif

#endif // NVTX_UTIL_HPP
