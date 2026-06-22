// GPU-native implementation of SolveTheta::profile.
//
// Mirrors the structure of the Fortran reference solve_theta_plan_many_cuda
// in TDMA/PaScaL_TDMA_F/examples/solve_theta.f90:
//   - All field/coefficient arrays live on the device for the entire run
//   - One H2D for theta at start, one D2H at end
//   - Each time step: build_RHS → boundary corrections → (build_LHS + batched
//     TDMA solve + permute back) per direction → update theta → ghost exchange
//   - The TDMA solver is called with device pointers (PaScaLTDMAManyCUDA::solve)
//
// Layout convention for the interior cube (ix*iy*iz where ix=nx-2 etc.):
//   d_rhs[ci] with ci = kk*iy*ix + jj*ix + ii,  ii=i-1, jj=j-1, kk=k-1
//   This is row-major (i,j,k) with i fastest, k slowest — matches the
//   full-grid indexing used by idx_ijk(i,j,k,nx,ny) = k*ny*nx + j*nx + i.

#include "solve_theta.hpp"
#include "tdma_backend_gpu.hpp"
#include "stencil_coeffs.hpp"
#include "index.hpp"
#include "timing_csv.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            std::fprintf(stderr,                                               \
                         "CUDA error %s at %s:%d: %s\n",                       \
                         cudaGetErrorName(_err), __FILE__, __LINE__,           \
                         cudaGetErrorString(_err));                            \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

namespace {

constexpr double D_PI = 3.14159265358979323846;

// Device-side stencil computation, mirrors stencil_coeffs.hpp
__device__ inline void d_stencil(double dt, double dd, int lb, int rb,
                                 double& a, double& b, double& c) {
    double base = dt / (2.0 * dd * dd);
    a = base * ( 1.0 + (5.0/3.0) * lb + (1.0/3.0) * rb);
    b = base * (-2.0 -        2.0 * lb -        2.0 * rb);
    c = base * ( 1.0 + (1.0/3.0) * lb + (5.0/3.0) * rb);
}

// Full-grid 3D indexer: theta[(k * ny + j) * nx + i]
__device__ inline std::size_t idx_full_d(int i, int j, int k, int nx, int ny) {
    return ((std::size_t)k * ny + (std::size_t)j) * nx + (std::size_t)i;
}

// =============================================================================
//  RHS computation — interior cells (i,j,k) in [1,nx-1) × [1,ny-1) × [1,nz-1)
// =============================================================================
__global__ void rhs_kernel(double* __restrict__ d_rhs,
                           const double* __restrict__ d_theta,
                           const double* __restrict__ dmx,
                           const double* __restrict__ dmy,
                           const double* __restrict__ dmz,
                           const int* __restrict__ x_lb, const int* __restrict__ x_rb,
                           const int* __restrict__ y_lb, const int* __restrict__ y_rb,
                           const int* __restrict__ z_lb, const int* __restrict__ z_rb,
                           const double* __restrict__ x_sub,
                           const double* __restrict__ y_sub,
                           const double* __restrict__ z_sub,
                           int nx, int ny, int nz, double dt) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int jj = blockIdx.y * blockDim.y + threadIdx.y;
    int kk = blockIdx.z * blockDim.z + threadIdx.z;
    int ix = nx - 2, iy = ny - 2, iz = nz - 2;
    if (ii >= ix || jj >= iy || kk >= iz) return;
    int i = ii + 1, j = jj + 1, k = kk + 1;

    double sxa, sxb, sxc;  d_stencil(dt, dmx[i], x_lb[i], x_rb[i], sxa, sxb, sxc);
    double sya, syb, syc;  d_stencil(dt, dmy[j], y_lb[j], y_rb[j], sya, syb, syc);
    double sza, szb, szc;  d_stencil(dt, dmz[k], z_lb[k], z_rb[k], sza, szb, szc);

    double t   = d_theta[idx_full_d(i,   j,   k,   nx, ny)];
    double tip = d_theta[idx_full_d(i+1, j,   k,   nx, ny)];
    double tim = d_theta[idx_full_d(i-1, j,   k,   nx, ny)];
    double tjp = d_theta[idx_full_d(i,   j+1, k,   nx, ny)];
    double tjm = d_theta[idx_full_d(i,   j-1, k,   nx, ny)];
    double tkp = d_theta[idx_full_d(i,   j,   k+1, nx, ny)];
    double tkm = d_theta[idx_full_d(i,   j,   k-1, nx, ny)];

    double src = dt * 3.0 * D_PI * D_PI * cos(D_PI*x_sub[i])
                                        * cos(D_PI*y_sub[j])
                                        * cos(D_PI*z_sub[k]);

    double rhs = (sxc * tip + sxa * tim)
               + (syc * tjp + sya * tjm)
               + (szc * tkp + sza * tkm)
               + (1.0 + sxb + syb + szb) * t
               + src;

    std::size_t ci = ((std::size_t)kk * iy + jj) * ix + ii;
    d_rhs[ci] = rhs;
}

// =============================================================================
//  Z-boundary correction
// =============================================================================
__global__ void z_boundary_kernel(double* __restrict__ d_rhs,
                                  const double* __restrict__ theta_z_left,
                                  const double* __restrict__ theta_z_right,
                                  const int* __restrict__ z_lb_flag,
                                  const int* __restrict__ z_rb_flag,
                                  const double* __restrict__ dmx,
                                  const double* __restrict__ dmy,
                                  const double* __restrict__ dmz,
                                  const int* __restrict__ x_lb, const int* __restrict__ x_rb,
                                  const int* __restrict__ y_lb, const int* __restrict__ y_rb,
                                  int nx, int ny, int nz, double dt) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int jj = blockIdx.y * blockDim.y + threadIdx.y;
    int ix = nx - 2, iy = ny - 2, iz = nz - 2;
    if (ii >= ix || jj >= iy) return;
    int i = ii + 1, j = jj + 1;

    double sxa, sxb, sxc;  d_stencil(dt, dmx[i], x_lb[i], x_rb[i], sxa, sxb, sxc);
    double sya, syb, syc;  d_stencil(dt, dmy[j], y_lb[j], y_rb[j], sya, syb, syc);

    // Left wall (k=1 interior cell touches k=0 wall via theta_z_left_sub)
    {
        double dz0 = dmz[0];
        double coef = (dt * 0.5 / (dz0*dz0)) * (1.0 + 5.0/3.0);
        double accum = 0.0;
        for (int dj = -1; dj <= 1; ++dj) {
            double cy = (dj == -1) ? -sya : (dj == 0) ? (1.0 - syb) : -syc;
            // theta_z_left_sub uses idx_ij(i, j+dj, nx) = (j+dj)*nx + i
            std::size_t off    = (std::size_t)(j+dj) * nx + i;
            std::size_t off_ip = (std::size_t)(j+dj) * nx + (i+1);
            std::size_t off_im = (std::size_t)(j+dj) * nx + (i-1);
            accum += coef * cy *
                     (-sxa        * theta_z_left[off_im]
                     + (1.0-sxb) * theta_z_left[off]
                     - sxc        * theta_z_left[off_ip])
                     * (double)z_lb_flag[1];
        }
        std::size_t ci = ((std::size_t)0 * iy + jj) * ix + ii;     // k=1 → kk=0
        d_rhs[ci] += accum;
    }
    // Right wall (k=nz-2 interior cell touches k=nz-1 wall)
    {
        double dzN = dmz[nz-1];
        double coef = (dt * 0.5 / (dzN*dzN)) * (1.0 + 5.0/3.0);
        double accum = 0.0;
        for (int dj = -1; dj <= 1; ++dj) {
            double cy = (dj == -1) ? -sya : (dj == 0) ? (1.0 - syb) : -syc;
            std::size_t off    = (std::size_t)(j+dj) * nx + i;
            std::size_t off_ip = (std::size_t)(j+dj) * nx + (i+1);
            std::size_t off_im = (std::size_t)(j+dj) * nx + (i-1);
            accum += coef * cy *
                     (-sxa        * theta_z_right[off_im]
                     + (1.0-sxb) * theta_z_right[off]
                     - sxc        * theta_z_right[off_ip])
                     * (double)z_rb_flag[nz-2];
        }
        std::size_t ci = ((std::size_t)(iz-1) * iy + jj) * ix + ii;
        d_rhs[ci] += accum;
    }
}

// =============================================================================
//  Y-boundary correction
// =============================================================================
__global__ void y_boundary_kernel(double* __restrict__ d_rhs,
                                  const double* __restrict__ theta_y_left,
                                  const double* __restrict__ theta_y_right,
                                  const int* __restrict__ y_lb_flag,
                                  const int* __restrict__ y_rb_flag,
                                  const double* __restrict__ dmx,
                                  const double* __restrict__ dmy,
                                  const int* __restrict__ x_lb, const int* __restrict__ x_rb,
                                  int nx, int ny, int nz, double dt) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int kk = blockIdx.y * blockDim.y + threadIdx.y;
    int ix = nx - 2, iy = ny - 2, iz = nz - 2;
    if (ii >= ix || kk >= iz) return;
    int i = ii + 1, k = kk + 1;

    double sxa, sxb, sxc; d_stencil(dt, dmx[i], x_lb[i], x_rb[i], sxa, sxb, sxc);

    // theta_y_*_sub uses idx_ik(i, k, nx) = k*nx + i
    std::size_t off    = (std::size_t)k * nx + i;
    std::size_t off_ip = (std::size_t)k * nx + (i+1);
    std::size_t off_im = (std::size_t)k * nx + (i-1);

    {
        double dy0 = dmy[0];
        double coef = (dt * 0.5 / (dy0*dy0)) * (1.0 + 5.0/3.0);
        std::size_t ci = ((std::size_t)kk * iy + 0) * ix + ii;     // j=1 → jj=0
        d_rhs[ci] += coef * (double)y_lb_flag[1] *
                     (-sxa * theta_y_left[off_im]
                     + (1.0-sxb) * theta_y_left[off]
                     - sxc * theta_y_left[off_ip]);
    }
    {
        double dyN = dmy[ny-1];
        double coef = (dt * 0.5 / (dyN*dyN)) * (1.0 + 5.0/3.0);
        std::size_t ci = ((std::size_t)kk * iy + (iy-1)) * ix + ii;
        d_rhs[ci] += coef * (double)y_rb_flag[ny-2] *
                     (-sxa * theta_y_right[off_im]
                     + (1.0-sxb) * theta_y_right[off]
                     - sxc * theta_y_right[off_ip]);
    }
}

// =============================================================================
//  X-boundary correction
// =============================================================================
__global__ void x_boundary_kernel(double* __restrict__ d_rhs,
                                  const double* __restrict__ theta_x_left,
                                  const double* __restrict__ theta_x_right,
                                  const int* __restrict__ x_lb_flag,
                                  const int* __restrict__ x_rb_flag,
                                  const double* __restrict__ dmx,
                                  int nx, int ny, int nz, double dt) {
    int jj = blockIdx.x * blockDim.x + threadIdx.x;
    int kk = blockIdx.y * blockDim.y + threadIdx.y;
    int ix = nx - 2, iy = ny - 2, iz = nz - 2;
    if (jj >= iy || kk >= iz) return;
    int j = jj + 1, k = kk + 1;

    // theta_x_*_sub uses idx_jk(j, k, ny) = k*ny + j
    std::size_t off = (std::size_t)k * ny + j;

    {
        double dx0 = dmx[0];
        double coef = (dt * 0.5 / (dx0*dx0)) * (1.0 + 5.0/3.0);
        std::size_t ci = ((std::size_t)kk * iy + jj) * ix + 0;     // i=1 → ii=0
        d_rhs[ci] += coef * (double)x_lb_flag[1] * theta_x_left[off];
    }
    {
        double dxN = dmx[nx-1];
        double coef = (dt * 0.5 / (dxN*dxN)) * (1.0 + 5.0/3.0);
        std::size_t ci = ((std::size_t)kk * iy + jj) * ix + (ix-1);
        d_rhs[ci] += coef * (double)x_rb_flag[nx-2] * theta_x_right[off];
    }
}

// =============================================================================
//  Build LHS for Z-direction sweep.
//  Layout: d_X[ ((kk * iy) + jj) * ix + ii ]  — same as d_rhs.
//  Treated by TDMA as [n_row=iz × n_sys=ix*iy], row-major.
// =============================================================================
__global__ void build_lhs_z_kernel(double* __restrict__ d_A,
                                   double* __restrict__ d_B,
                                   double* __restrict__ d_C,
                                   double* __restrict__ d_D,
                                   const double* __restrict__ d_rhs,
                                   const double* __restrict__ dmz,
                                   const int* __restrict__ z_lb,
                                   const int* __restrict__ z_rb,
                                   int ix, int iy, int iz, double dt) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int jj = blockIdx.y * blockDim.y + threadIdx.y;
    int kk = blockIdx.z * blockDim.z + threadIdx.z;
    if (ii >= ix || jj >= iy || kk >= iz) return;
    int k = kk + 1;

    double sa, sb, sc;  d_stencil(dt, dmz[k], z_lb[k], z_rb[k], sa, sb, sc);

    std::size_t off = ((std::size_t)kk * iy + jj) * ix + ii;
    d_A[off] = -sa;
    d_B[off] = 1.0 - sb;
    d_C[off] = -sc;
    d_D[off] = d_rhs[off];
}

// Copy Z-solution back into rhs (same layout as the build).
__global__ void copy_z_to_rhs(double* __restrict__ d_rhs,
                              const double* __restrict__ d_D,
                              std::size_t n) {
    std::size_t idx = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d_rhs[idx] = d_D[idx];
}

// =============================================================================
//  Build LHS for Y-direction sweep.
//  Layout: d_X[ ((jj * iz) + kk) * ix + ii ] — j becomes the row dimension.
//  Treated by TDMA as [n_row=iy × n_sys=iz*ix], row-major.
// =============================================================================
__global__ void build_lhs_y_kernel(double* __restrict__ d_A,
                                   double* __restrict__ d_B,
                                   double* __restrict__ d_C,
                                   double* __restrict__ d_D,
                                   const double* __restrict__ d_rhs,
                                   const double* __restrict__ dmy,
                                   const int* __restrict__ y_lb,
                                   const int* __restrict__ y_rb,
                                   int ix, int iy, int iz, double dt) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int jj = blockIdx.y * blockDim.y + threadIdx.y;
    int kk = blockIdx.z * blockDim.z + threadIdx.z;
    if (ii >= ix || jj >= iy || kk >= iz) return;
    int j = jj + 1;

    double sa, sb, sc;  d_stencil(dt, dmy[j], y_lb[j], y_rb[j], sa, sb, sc);

    std::size_t off_y   = ((std::size_t)jj * iz + kk) * ix + ii;
    std::size_t off_rhs = ((std::size_t)kk * iy + jj) * ix + ii;
    d_A[off_y] = -sa;
    d_B[off_y] = 1.0 - sb;
    d_C[off_y] = -sc;
    d_D[off_y] = d_rhs[off_rhs];
}

// Copy Y-solution back into rhs (transposing j↔k).
__global__ void copy_y_to_rhs(double* __restrict__ d_rhs,
                              const double* __restrict__ d_D,
                              int ix, int iy, int iz) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int jj = blockIdx.y * blockDim.y + threadIdx.y;
    int kk = blockIdx.z * blockDim.z + threadIdx.z;
    if (ii >= ix || jj >= iy || kk >= iz) return;
    std::size_t off_y   = ((std::size_t)jj * iz + kk) * ix + ii;
    std::size_t off_rhs = ((std::size_t)kk * iy + jj) * ix + ii;
    d_rhs[off_rhs] = d_D[off_y];
}

// Build X-LHS. d_rhs has ii fastest, d_X has jj fastest — stage d_rhs through
// a shared tile so reads coalesce on ii and writes coalesce on jj.
__global__ void build_lhs_x_kernel(double* __restrict__ d_A,
                                   double* __restrict__ d_B,
                                   double* __restrict__ d_C,
                                   double* __restrict__ d_D,
                                   const double* __restrict__ d_rhs,
                                   const double* __restrict__ dmx,
                                   const int* __restrict__ x_lb,
                                   const int* __restrict__ x_rb,
                                   int ix, int iy, int iz, double dt) {
    constexpr int TILE = 32;
    __shared__ double tile[TILE][TILE + 1];

    int ii_b = blockIdx.x * TILE;
    int jj_b = blockIdx.y * TILE;
    int kk   = blockIdx.z;
    if (kk >= iz) return;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    {
        int ii_g = ii_b + tx;
        int jj_g = jj_b + ty;
        if (ii_g < ix && jj_g < iy) {
            tile[ty][tx] = d_rhs[((std::size_t)kk * iy + jj_g) * ix + ii_g];
        }
    }
    __syncthreads();

    {
        int jj_g = jj_b + tx;
        int ii_g = ii_b + ty;
        if (ii_g < ix && jj_g < iy) {
            int i = ii_g + 1;
            double sa, sb, sc;
            d_stencil(dt, dmx[i], x_lb[i], x_rb[i], sa, sb, sc);
            std::size_t off_x = ((std::size_t)ii_g * iz + kk) * iy + jj_g;
            d_A[off_x] = -sa;
            d_B[off_x] = 1.0 - sb;
            d_C[off_x] = -sc;
            d_D[off_x] = tile[tx][ty];
        }
    }
}

// Final update: theta(i,j,k) = D_x(i,k,j).
__global__ void update_theta_kernel(double* __restrict__ d_theta,
                                    const double* __restrict__ d_D,
                                    int ix, int iy, int iz,
                                    int nx, int ny) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int jj = blockIdx.y * blockDim.y + threadIdx.y;
    int kk = blockIdx.z * blockDim.z + threadIdx.z;
    if (ii >= ix || jj >= iy || kk >= iz) return;
    int i = ii + 1, j = jj + 1, k = kk + 1;
    std::size_t off_x = ((std::size_t)ii * iz + kk) * iy + jj;
    d_theta[idx_full_d(i, j, k, nx, ny)] = d_D[off_x];
}

// =============================================================================
//  Helpers to upload host vectors / push int arrays
// =============================================================================
template <class T>
T* alloc_and_copy(const std::vector<T>& src) {
    T* p = nullptr;
    std::size_t bytes = sizeof(T) * src.size();
    CUDA_CHECK(cudaMalloc(&p, bytes));
    CUDA_CHECK(cudaMemcpy(p, src.data(), bytes, cudaMemcpyHostToDevice));
    return p;
}

inline dim3 grid3(int nx, int ny, int nz, dim3 block) {
    return dim3((nx + block.x - 1) / block.x,
                (ny + block.y - 1) / block.y,
                (nz + block.z - 1) / block.z);
}

inline dim3 grid2(int nx, int ny, dim3 block) {
    return dim3((nx + block.x - 1) / block.x,
                (ny + block.y - 1) / block.y, 1);
}

} // namespace

// ===========================================================================
//  SolveTheta::profile  — GPU-native time-step loop
// ===========================================================================
SolveTheta::SolveTheta(const GlobalParams& params,
                       const MPITopology& topo,
                       MPISubdomain& sub)
    : params_(params), topo_(topo), sub_(sub) {}

// Currently `run()` and `profile()` share the same kernel sequence — the
// per-step timing dumps that distinguished them in the CPU code are not
// active on the GPU path yet.
void SolveTheta::run(std::vector<double>& theta) { profile(theta); }

void SolveTheta::profile(std::vector<double>& theta) {
    int nx_full = sub_.nx_sub + 1;     // matches host nx1
    int ny_full = sub_.ny_sub + 1;
    int nz_full = sub_.nz_sub + 1;
    int ix = nx_full - 2;              // interior count
    int iy = ny_full - 2;
    int iz = nz_full - 2;
    int max_iter  = params_.Nt;
    int n_warmup  = params_.Nt_warmup;
    int n_timing  = max_iter - n_warmup;
    double dt     = params_.dt;

    auto cx = topo_.commX();
    auto cy = topo_.commY();
    auto cz = topo_.commZ();

    int my_rank;  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if (my_rank == 0) {
        std::cout << "[Tmax] = " << dt * max_iter
                  << " | [max_iter] = " << max_iter << "\n";
        std::cout << "[nx] = " << ix << " | [ny] = " << iy
                  << " | [nz] = " << iz << "\n";
        double beta = dt / 2.0 / (sub_.dmz_sub[1] * sub_.dmz_sub[1]);
        std::cout << "[rho] = " << beta / (1.0 + 2.0 * beta) << "\n";
    }

    // -------------------------------------------------------------------
    //  Persistent device buffers
    // -------------------------------------------------------------------
    std::size_t full_n  = (std::size_t)nx_full * ny_full * nz_full;
    std::size_t inner_n = (std::size_t)ix * iy * iz;

    // Pre-allocate the 12 contiguous ghost-cell send/recv buffers on the
    // device.  Replaces the old derived-datatype-on-device-pointer path —
    // CUDA-aware MPI is far faster on contiguous device buffers.
    sub_.allocGhostBufsDevice();

    double* d_theta = nullptr; CUDA_CHECK(cudaMalloc(&d_theta, sizeof(double) * full_n));
    double* d_rhs   = nullptr; CUDA_CHECK(cudaMalloc(&d_rhs,   sizeof(double) * inner_n));
    double* d_A     = nullptr; CUDA_CHECK(cudaMalloc(&d_A,     sizeof(double) * inner_n));
    double* d_B     = nullptr; CUDA_CHECK(cudaMalloc(&d_B,     sizeof(double) * inner_n));
    double* d_C     = nullptr; CUDA_CHECK(cudaMalloc(&d_C,     sizeof(double) * inner_n));
    double* d_D     = nullptr; CUDA_CHECK(cudaMalloc(&d_D,     sizeof(double) * inner_n));

    // Mesh / boundary arrays — one-time upload
    double* d_dmx     = alloc_and_copy(sub_.dmx_sub);
    double* d_dmy     = alloc_and_copy(sub_.dmy_sub);
    double* d_dmz     = alloc_and_copy(sub_.dmz_sub);
    double* d_x_sub   = alloc_and_copy(sub_.x_sub);
    double* d_y_sub   = alloc_and_copy(sub_.y_sub);
    double* d_z_sub   = alloc_and_copy(sub_.z_sub);
    int*    d_x_lb    = alloc_and_copy(sub_.theta_x_left_index);
    int*    d_x_rb    = alloc_and_copy(sub_.theta_x_right_index);
    int*    d_y_lb    = alloc_and_copy(sub_.theta_y_left_index);
    int*    d_y_rb    = alloc_and_copy(sub_.theta_y_right_index);
    int*    d_z_lb    = alloc_and_copy(sub_.theta_z_left_index);
    int*    d_z_rb    = alloc_and_copy(sub_.theta_z_right_index);
    double* d_th_xL   = alloc_and_copy(sub_.theta_x_left_sub);
    double* d_th_xR   = alloc_and_copy(sub_.theta_x_right_sub);
    double* d_th_yL   = alloc_and_copy(sub_.theta_y_left_sub);
    double* d_th_yR   = alloc_and_copy(sub_.theta_y_right_sub);
    double* d_th_zL   = alloc_and_copy(sub_.theta_z_left_sub);
    double* d_th_zR   = alloc_and_copy(sub_.theta_z_right_sub);

    // theta H2D — once
    CUDA_CHECK(cudaMemcpy(d_theta, theta.data(), sizeof(double) * full_n,
                          cudaMemcpyHostToDevice));

    // -------------------------------------------------------------------
    //  Solver plans (one per direction).  Lifetime spans all time steps.
    //  Backend (filtered | pascal) is chosen by params_.tdma_backend.
    // -------------------------------------------------------------------
    const auto kind = TdmaBackendGPU::parse(params_.tdma_backend);
    TdmaBackendGPU solver_z(kind, ix * iy, iz,
                            cz.myrank, cz.nprocs, cz.comm,
                            cz.west_rank, cz.east_rank, dt);
    TdmaBackendGPU solver_y(kind, ix * iz, iy,
                            cy.myrank, cy.nprocs, cy.comm,
                            cy.west_rank, cy.east_rank, dt);
    TdmaBackendGPU solver_x(kind, iy * iz, ix,
                            cx.myrank, cx.nprocs, cx.comm,
                            cx.west_rank, cx.east_rank, dt);
    if (my_rank == 0) {
        std::cout << "[backend] = " << params_.tdma_backend << "\n";
    }

    dim3 b3(8, 4, 4);
    dim3 g3 = grid3(ix, iy, iz, b3);

    constexpr int LHS_X_TILE = 32;
    dim3 b3_lhs_x(LHS_X_TILE, LHS_X_TILE, 1);
    dim3 g3_lhs_x((ix + LHS_X_TILE - 1) / LHS_X_TILE,
                  (iy + LHS_X_TILE - 1) / LHS_X_TILE,
                  iz);
    dim3 b2(16, 16, 1);
    dim3 g_xy = grid2(ix, iy, b2);
    dim3 g_xz = grid2(ix, iz, b2);
    dim3 g_yz = grid2(iy, iz, b2);

    // -------------------------------------------------------------------
    //  Time-step loop  (per-step, per-rank, per-event timing → long CSV)
    //  Events: rhs, solve_z, solve_y, solve_x, etc, comm
    //  Timing CSV is written only for scaling tests (option != "order").
    // -------------------------------------------------------------------
    const bool do_timing = (params_.option != "order");
    const std::vector<std::string> event_names =
        {"rhs", "solve_z", "solve_y", "solve_x", "etc", "comm"};
    const int n_events = (int)event_names.size();
    if (do_timing)
        timing_csv::timing_init(n_events, n_timing, MPI_COMM_WORLD);
    std::vector<double> local_times(n_events, 0.0);

    for (int t_step = 0; t_step < max_iter; ++t_step) {
        double t0, t1;

        // --- comm (ghost cell exchange — at start of step) ---
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        sub_.ghostcellUpdateDevice(d_theta, cx, cy, cz);
        CUDA_CHECK(cudaDeviceSynchronize());
        t1 = MPI_Wtime();
        local_times[5] = t1 - t0;

        // --- etc: backend refresh (filtered backend needs eps refresh per step) ---
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        solver_z.set_eps_constant(dt);
        solver_y.set_eps_constant(dt);
        solver_x.set_eps_constant(dt);
        CUDA_CHECK(cudaDeviceSynchronize());
        t1 = MPI_Wtime();
        local_times[4] = t1 - t0;

        // --- RHS ---
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        rhs_kernel<<<g3, b3>>>(d_rhs, d_theta,
                               d_dmx, d_dmy, d_dmz,
                               d_x_lb, d_x_rb, d_y_lb, d_y_rb, d_z_lb, d_z_rb,
                               d_x_sub, d_y_sub, d_z_sub,
                               nx_full, ny_full, nz_full, dt);
        CUDA_CHECK(cudaDeviceSynchronize());
        t1 = MPI_Wtime();
        local_times[0] = t1 - t0;

        // --- Z direction (boundary correction + build_LHS + TDMA + copy back) ---
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        z_boundary_kernel<<<g_xy, b2>>>(d_rhs,
                                        d_th_zL, d_th_zR,
                                        d_z_lb, d_z_rb,
                                        d_dmx, d_dmy, d_dmz,
                                        d_x_lb, d_x_rb, d_y_lb, d_y_rb,
                                        nx_full, ny_full, nz_full, dt);
        build_lhs_z_kernel<<<g3, b3>>>(d_A, d_B, d_C, d_D, d_rhs,
                                       d_dmz, d_z_lb, d_z_rb, ix, iy, iz, dt);
        solver_z.set_rho_device(d_A, d_B, d_C);
        solver_z.solve(d_A, d_B, d_C, d_D);
        {
            const int block_lin = 256;
            const int grid_lin  = (int)((inner_n + block_lin - 1) / block_lin);
            copy_z_to_rhs<<<grid_lin, block_lin>>>(d_rhs, d_D, inner_n);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        t1 = MPI_Wtime();
        local_times[1] = t1 - t0;

        // --- Y direction ---
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        y_boundary_kernel<<<g_xz, b2>>>(d_rhs,
                                        d_th_yL, d_th_yR,
                                        d_y_lb, d_y_rb,
                                        d_dmx, d_dmy,
                                        d_x_lb, d_x_rb,
                                        nx_full, ny_full, nz_full, dt);
        build_lhs_y_kernel<<<g3, b3>>>(d_A, d_B, d_C, d_D, d_rhs,
                                       d_dmy, d_y_lb, d_y_rb, ix, iy, iz, dt);
        solver_y.set_rho_device(d_A, d_B, d_C);
        solver_y.solve(d_A, d_B, d_C, d_D);
        copy_y_to_rhs<<<g3, b3>>>(d_rhs, d_D, ix, iy, iz);
        CUDA_CHECK(cudaDeviceSynchronize());
        t1 = MPI_Wtime();
        local_times[2] = t1 - t0;

        // --- X direction ---
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        x_boundary_kernel<<<g_yz, b2>>>(d_rhs,
                                        d_th_xL, d_th_xR,
                                        d_x_lb, d_x_rb,
                                        d_dmx,
                                        nx_full, ny_full, nz_full, dt);
        build_lhs_x_kernel<<<g3_lhs_x, b3_lhs_x>>>(d_A, d_B, d_C, d_D, d_rhs,
                                       d_dmx, d_x_lb, d_x_rb, ix, iy, iz, dt);
        solver_x.set_rho_device(d_A, d_B, d_C);
        solver_x.solve(d_A, d_B, d_C, d_D);
        update_theta_kernel<<<g3, b3>>>(d_theta, d_D, ix, iy, iz, nx_full, ny_full);
        CUDA_CHECK(cudaDeviceSynchronize());
        t1 = MPI_Wtime();
        local_times[3] = t1 - t0;

        // Skip warmup steps; record only the timed phase.
        if (do_timing && t_step >= n_warmup) {
            timing_csv::timing_record(t_step - n_warmup, local_times, MPI_COMM_WORLD);
        }
    }

    // -------------------------------------------------------------------
    //  Write long-format CSV (rank 0 only) — scaling tests only.
    //  Path:  results/timing_<N>_<npx><npy><npz>_<backend>.csv
    //  Meta:  # grid=NxNxN, np=NP (npx,npy,npz), dt=..., Nt=..., solver_kind=<backend>
    // -------------------------------------------------------------------
    if (do_timing) {
        char meta[256];
        std::snprintf(meta, sizeof(meta),
                      "grid=%dx%dx%d, np=%d (%d,%d,%d), dt=%10.3E, Nt=%d, solver_kind=%s",
                      params_.nx, params_.ny, params_.nz,
                      cx.nprocs * cy.nprocs * cz.nprocs,
                      params_.np_dim[0], params_.np_dim[1], params_.np_dim[2],
                      dt, max_iter, params_.tdma_backend.c_str());

        char fn[256];
        const char* env_path = std::getenv("TIMING_CSV");
        if (env_path && env_path[0] != '\0') {
            std::snprintf(fn, sizeof(fn), "%s", env_path);
        } else {
            std::snprintf(fn, sizeof(fn), "results/timing_%d_%d%d%d_%s.csv",
                          params_.nx,
                          params_.np_dim[0], params_.np_dim[1], params_.np_dim[2],
                          params_.tdma_backend.c_str());
        }

        timing_csv::timing_save_csv(fn, event_names, meta, MPI_COMM_WORLD);
    }
    timing_csv::timing_cleanup();

    // -------------------------------------------------------------------
    //  D2H once
    // -------------------------------------------------------------------
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(theta.data(), d_theta, sizeof(double) * full_n,
                          cudaMemcpyDeviceToHost));

    // -------------------------------------------------------------------
    //  Free
    // -------------------------------------------------------------------
    auto safe_free = [](void*& p) { if (p) { cudaFree(p); p = nullptr; } };
    safe_free((void*&)d_theta); safe_free((void*&)d_rhs);
    safe_free((void*&)d_A); safe_free((void*&)d_B); safe_free((void*&)d_C); safe_free((void*&)d_D);
    safe_free((void*&)d_dmx); safe_free((void*&)d_dmy); safe_free((void*&)d_dmz);
    safe_free((void*&)d_x_sub); safe_free((void*&)d_y_sub); safe_free((void*&)d_z_sub);
    safe_free((void*&)d_x_lb); safe_free((void*&)d_x_rb);
    safe_free((void*&)d_y_lb); safe_free((void*&)d_y_rb);
    safe_free((void*&)d_z_lb); safe_free((void*&)d_z_rb);
    safe_free((void*&)d_th_xL); safe_free((void*&)d_th_xR);
    safe_free((void*&)d_th_yL); safe_free((void*&)d_th_yR);
    safe_free((void*&)d_th_zL); safe_free((void*&)d_th_zR);
    sub_.freeGhostBufsDevice();
}
