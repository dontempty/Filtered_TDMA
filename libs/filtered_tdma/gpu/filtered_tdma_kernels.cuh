#ifndef FILTERED_TDMA_KERNELS_CUH
#define FILTERED_TDMA_KERNELS_CUH

// Private CUDA kernel definitions shared between filtered_tdma_cuda.cu (production)
// and filtered_tdma_profile_cuda.cu (profiling). Each TU gets its own static copy
// of the device code; no -rdc required.

#include <cuda_runtime.h>
#include <cstddef>
#include <cmath>

static constexpr int FTDMA_BLOCK_SYS = 256;

static inline dim3 ftdma_grid1D(int n) {
    return dim3((n + FTDMA_BLOCK_SYS - 1) / FTDMA_BLOCK_SYS);
}

__device__ static inline double ftdma_warp_max(double v) {
    for (int s = 16; s > 0; s >>= 1)
        v = fmax(v, __shfl_down_sync(0xffffffffu, v, s));
    return v;
}

__device__ static inline void ftdma_reduce3_max(double* s_a, double* s_b, double* s_c, int tid) {
    for (int s = FTDMA_BLOCK_SYS / 2; s >= 32; s >>= 1) {
        if (tid < s) {
            s_a[tid] = fmax(s_a[tid], s_a[tid + s]);
            s_b[tid] = fmax(s_b[tid], s_b[tid + s]);
            s_c[tid] = fmax(s_c[tid], s_c[tid + s]);
        }
        __syncthreads();
    }
    if (tid < 32) {
        double a = ftdma_warp_max(s_a[tid]);
        double b = ftdma_warp_max(s_b[tid]);
        double c = ftdma_warp_max(s_c[tid]);
        if (tid == 0) { s_a[0] = a; s_b[0] = b; s_c[0] = c; }
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
//  Kernels
// ---------------------------------------------------------------------------

static __global__ void k_set_rho(double* __restrict__ d_A_rho,
                                  double* __restrict__ d_C_rho,
                                  const double* __restrict__ d_A,
                                  const double* __restrict__ d_B,
                                  const double* __restrict__ d_C,
                                  int n_sys, int n_row) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_row + 1) return;
    if (k < n_row) {
        double bk = __ldg(&d_B[(std::size_t)k * n_sys]);
        if (bk != 0.0) {
            d_A_rho[k] = fabs(__ldg(&d_A[(std::size_t)k * n_sys]) / bk);
            d_C_rho[k] = fabs(__ldg(&d_C[(std::size_t)k * n_sys]) / bk);
        } else {
            d_A_rho[k] = 0.0;
            d_C_rho[k] = 0.0;
        }
    } else {
        d_A_rho[k] = 0.0;
        d_C_rho[k] = 0.0;
    }
}

static __global__ __launch_bounds__(FTDMA_BLOCK_SYS)
void k_fwd_pass_v1(double* A, double* B, double* C, double* D,
                   int n_sys, int n_row) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;

    double r0 = 1.0 / B[i];
    A[i] *= r0;  C[i] *= r0;  D[i] *= r0;

    std::size_t off1 = (std::size_t)1 * n_sys + i;
    double r1 = 1.0 / B[off1];
    A[off1] *= r1;  C[off1] *= r1;  D[off1] *= r1;

    double a_prev = A[off1], c_prev = C[off1], d_prev = D[off1];

    for (int j = 2; j < n_row; ++j) {
        std::size_t off_j = (std::size_t)j * n_sys + i;
        double Aj = A[off_j], Bj = B[off_j], Cj = C[off_j], Dj = D[off_j];
        double inv  = 1.0 / (Bj - Aj * c_prev);
        double d_new =  inv * (Dj - Aj * d_prev);
        double c_new =  inv * Cj;
        double a_new = -inv * Aj * a_prev;
        D[off_j] = d_new;  C[off_j] = c_new;  A[off_j] = a_new;
        a_prev = a_new;  c_prev = c_new;  d_prev = d_new;
    }
}

static __global__ __launch_bounds__(FTDMA_BLOCK_SYS)
void k_bwd_pass_v1(double* A, double* C, double* D,
                   int n_sys, int n_row) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;

    std::size_t off_nm2 = (std::size_t)(n_row - 2) * n_sys + i;
    double d_next = D[off_nm2], a_next = A[off_nm2], c_next = C[off_nm2];

    for (int j = n_row - 3; j >= 1; --j) {
        std::size_t off_j = (std::size_t)j * n_sys + i;
        double Cj   = C[off_j];
        double d_new = D[off_j] - Cj * d_next;
        double a_new = A[off_j] - Cj * a_next;
        double c_new = -Cj * c_next;
        D[off_j] = d_new;  A[off_j] = a_new;  C[off_j] = c_new;
        d_next = d_new;  a_next = a_new;  c_next = c_new;
    }
    double C0 = C[i], D0_val = D[i];
    double r  = 1.0 / (1.0 - a_next * C0);
    D[i] = r * (D0_val - C0 * d_next);
    A[i] = r * A[i];
}

static __global__ __launch_bounds__(FTDMA_BLOCK_SYS)
void k_fwd_pass_v2(double* A, double* B, double* C, double* D,
                   int n_sys, int n_row, const int* __restrict__ d_J) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;
    const int J = __ldg(d_J);

    double r0 = 1.0 / B[i];
    A[i] *= r0;  C[i] *= r0;  D[i] *= r0;

    std::size_t off1 = (std::size_t)1 * n_sys + i;
    double r1 = 1.0 / B[off1];
    A[off1] *= r1;  C[off1] *= r1;  D[off1] *= r1;

    double a_prev = A[off1], c_prev = C[off1], d_prev = D[off1];

    for (int j = 2; j <= J; ++j) {
        std::size_t off_j = (std::size_t)j * n_sys + i;
        double Aj = A[off_j], Bj = B[off_j], Cj = C[off_j], Dj = D[off_j];
        double inv  = 1.0 / (Bj - Aj * c_prev);
        double d_new =  inv * (Dj - Aj * d_prev);
        double c_new =  inv * Cj;
        double a_new = -inv * Aj * a_prev;
        D[off_j] = d_new;  C[off_j] = c_new;  A[off_j] = a_new;
        a_prev = a_new;  c_prev = c_new;  d_prev = d_new;
    }
    for (int j = J + 1; j < n_row; ++j) {
        std::size_t off_j = (std::size_t)j * n_sys + i;
        double Aj = A[off_j], Bj = B[off_j], Cj = C[off_j], Dj = D[off_j];
        double inv  = 1.0 / (Bj - Aj * c_prev);
        double d_new = inv * (Dj - Aj * d_prev);
        double c_new = inv * Cj;
        D[off_j] = d_new;  C[off_j] = c_new;
        c_prev = c_new;  d_prev = d_new;
    }
}

static __global__ __launch_bounds__(FTDMA_BLOCK_SYS)
void k_bwd_pass_v2(double* A, double* C, double* D,
                   int n_sys, int n_row, const int* __restrict__ d_J) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;
    const int J  = __ldg(d_J);
    const int lo = (n_row - 2) - J;

    std::size_t off_nm2 = (std::size_t)(n_row - 2) * n_sys + i;
    double d_next = D[off_nm2];
    double c_next = C[off_nm2];
    double a_next = (J >= 2) ? A[(std::size_t)J * n_sys + i] : 0.0;

    for (int j = n_row - 3; j >= 1; --j) {
        std::size_t off_j = (std::size_t)j * n_sys + i;
        double Cj = C[off_j];
        double d_new = D[off_j] - Cj * d_next;
        D[off_j] = d_new;
        d_next = d_new;
        if (J >= 2 && j < J) {
            double a_new = A[off_j] - Cj * a_next;
            A[off_j] = a_new;
            a_next = a_new;
        }
        if (j > lo) {
            double c_new = -Cj * c_next;
            C[off_j] = c_new;
            c_next = c_new;
        }
    }
    double A1    = A[(std::size_t)1 * n_sys + i];
    double C0    = C[i];
    double D0_val = D[i];
    double D1    = D[(std::size_t)1 * n_sys + i];
    double r     = 1.0 / (1.0 - A1 * C0);
    D[i] = r * (D0_val - C0 * D1);
    A[i] = r * A[i];
}

static __global__ __launch_bounds__(FTDMA_BLOCK_SYS, 4)
void k_final_pass(const double* A, const double* C, double* D,
                  const double* D0, const double* DN,
                  int n_sys, int n_row, const int* __restrict__ d_J) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;

    const int J = __ldg(d_J);
    double d0 = D0[i];
    double dn = DN[i];
    int jL = J < n_row - 2 ? J : n_row - 2;
    int jR = (n_row - 1) - J;
    if (jR < 1) jR = 1;

    for (int j = 1; j <= jL; ++j) {
        std::size_t off = (std::size_t)j * n_sys + i;
        D[off] -= A[off] * d0;
    }
    for (int j = jR; j < n_row - 1; ++j) {
        std::size_t off = (std::size_t)j * n_sys + i;
        D[off] -= C[off] * dn;
    }
}

static __global__ void k_cal_J_rhs_bound(int* __restrict__ d_J,
                                          const double* __restrict__ d_A_rho,
                                          const double* __restrict__ d_C_rho,
                                          const double* __restrict__ D0,
                                          const double* __restrict__ DN,
                                          int n_sys, int n_row, double eps) {
    __shared__ double s_rho[FTDMA_BLOCK_SYS];
    __shared__ double s_D0 [FTDMA_BLOCK_SYS];
    __shared__ double s_DN [FTDMA_BLOCK_SYS];
    int tid = threadIdx.x;

    double my_rho = 0.0;
    for (int k = 2 + tid; k < n_row - 1; k += blockDim.x)
        my_rho = fmax(my_rho, fmax(fabs(d_A_rho[k]), fabs(d_C_rho[k])));

    double my_D0 = 0.0, my_DN = 0.0;
    for (int i = tid; i < n_sys; i += blockDim.x) {
        my_D0 = fmax(my_D0, fabs(D0[i]));
        my_DN = fmax(my_DN, fabs(DN[i]));
    }

    s_rho[tid] = my_rho; s_D0[tid] = my_D0; s_DN[tid] = my_DN;
    __syncthreads();
    ftdma_reduce3_max(s_rho, s_D0, s_DN, tid);

    if (tid == 0) {
        double rho   = s_rho[0];
        double max_b = fmax(s_D0[0], s_DN[0]);
        int J;
        if (rho == 0.0 || rho >= 0.5) {
            J = n_row - 1;
        } else {
            double lambda_p = (1.0 + sqrt(1.0 - 4.0 * rho * rho)) * 0.5;
            double q = rho / lambda_p;
            double B = q * (2.0 + q) / ((1.0 - q) * (1.0 - 2.0 * rho)) * max_b;
            int Jcomp = (int)(log(eps / B) / log(q)) + 1;
            J = (Jcomp < n_row - 1) ? Jcomp : n_row - 1;
            if (J < 0) J = 0;
        }
        *d_J = J;
    }
}

static __global__ void k_pack_lastrow(double* __restrict__ dst,
                                       const double* __restrict__ C,
                                       const double* __restrict__ D,
                                       int n_sys, int n_row) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;
    std::size_t last = (std::size_t)(n_row - 1) * n_sys;
    dst[i]         = C[last + i];
    dst[n_sys + i] = D[last + i];
}

static __global__ void k_pack_row0(double* __restrict__ dst,
                                    const double* __restrict__ A,
                                    const double* __restrict__ D,
                                    int n_sys) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;
    dst[i]         = A[i];
    dst[n_sys + i] = D[i];
}

static __global__ void k_solve_both(double* __restrict__ D,
                                     const double* __restrict__ A,
                                     const double* __restrict__ C,
                                     const double* __restrict__ C_left,
                                     const double* __restrict__ D_left,
                                     const double* __restrict__ A_right,
                                     const double* __restrict__ D0_right,
                                     int has_left, int has_right, int n_sys, int n_row) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;
    if (has_left) {
        double A0 = A[i], D0 = D[i];
        D[i] = (D0 - A0 * D_left[i]) / (1.0 - C_left[i] * A0);
    }
    if (has_right) {
        std::size_t off = (std::size_t)(n_row - 1) * n_sys + i;
        double CN = C[off], DN = D[off];
        D[off] = (DN - CN * D0_right[i]) / (1.0 - A_right[i] * CN);
    }
}

#endif // FILTERED_TDMA_KERNELS_CUH
