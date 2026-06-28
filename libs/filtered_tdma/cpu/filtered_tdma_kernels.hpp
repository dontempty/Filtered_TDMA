#ifndef FILTERED_TDMA_KERNELS_HPP
#define FILTERED_TDMA_KERNELS_HPP

// Shared CPU "kernels" (algorithm steps) for the Filtered TDMA solves.
//
// These are the reusable building blocks called by both the production solves
// in filtered_tdma.cpp and the per-phase timed solves in
// filtered_tdma_profile.cpp. Each is `static inline`, so every translation unit
// that includes this header gets its own internal-linkage copy — no link
// conflicts with the pascal_tdma library.
//
// Every step takes the matrix/RHS as raw base pointers (A, B, C, D) and derives
// the row offsets it needs internally — mirroring the GPU kernels, which take
// d_A/d_B/d_C/d_D. The base pointers are marked `__restrict`: A, B, C and D are
// four distinct arrays, so this is true, and it is REQUIRED for correctness, not
// just speed. nvc++ 25.11's auto-vectorizer otherwise miscompiles the strided
// forward/backward sweeps (verified: `-O0` or `-Mnovect` give correct results,
// the vectorized code without the aliasing info does not). With `__restrict` the
// vectorizer disambiguates the columns of A/B/C/D and emits correct SIMD code.
//
// The MPI neighbour exchange is NOT here: it needs the solver's MPI plan
// (datatypes, ranks, recv buffers) and lives as FilteredTDMA::exchange_interfaces_().

#include <cstddef>
#include <cmath>
#include <algorithm>

namespace ftdma_kernel {

// ---------------------------------------------------------------------------
//  Step 1: Forward elimination (normalize rows 0,1 then eliminate downward)
// ---------------------------------------------------------------------------

// v1 — full update (D, C, A) for every row j=2..n_row-1.
[[gnu::always_inline]] static inline
void fwd_elim_v1(double* __restrict A, double* __restrict B,
                 double* __restrict C, double* __restrict D,
                 int n_sys, int n_row) {
    int i, j;
    double* A1 = A + (std::size_t)1 * n_sys;
    double* B1 = B + (std::size_t)1 * n_sys;
    double* C1 = C + (std::size_t)1 * n_sys;
    double* D1 = D + (std::size_t)1 * n_sys;
    #pragma omp simd
    for (i = 0; i < n_sys; ++i) {
        double r0 = 1.0 / B[i];
        A[i] *= r0; C[i] *= r0; D[i] *= r0;
        double r1 = 1.0 / B1[i];
        A1[i] *= r1; C1[i] *= r1; D1[i] *= r1;
    }
    double* Ajm = A1, *Cjm = C1, *Djm = D1;
    double* Aj = A1 + n_sys, *Bj = B1 + n_sys;
    double* Cj = C1 + n_sys, *Dj = D1 + n_sys;
    for (j = 2; j < n_row; ++j,
         Ajm = Aj, Cjm = Cj, Djm = Dj,
         Aj += n_sys, Bj += n_sys, Cj += n_sys, Dj += n_sys) {
        #pragma omp simd
        for (i = 0; i < n_sys; ++i) {
            double inv = 1.0 / (Bj[i] - Aj[i] * Cjm[i]);
            Dj[i] =  inv * (Dj[i] - Aj[i] * Djm[i]);
            Cj[i] =  inv * Cj[i];
            Aj[i] = -inv * Aj[i] * Ajm[i];
        }
    }
}

// v2 — full update for j=2..J, then skip-A update for j=J+1..n_row-1.
[[gnu::always_inline]] static inline
void fwd_elim_v2(double* __restrict A, double* __restrict B,
                 double* __restrict C, double* __restrict D,
                 int n_sys, int n_row, int J) {
    int i, j;
    double* A1 = A + (std::size_t)1 * n_sys;
    double* B1 = B + (std::size_t)1 * n_sys;
    double* C1 = C + (std::size_t)1 * n_sys;
    double* D1 = D + (std::size_t)1 * n_sys;
    #pragma omp simd
    for (i = 0; i < n_sys; ++i) {
        double r0 = 1.0 / B[i];
        A[i] *= r0; C[i] *= r0; D[i] *= r0;
        double r1 = 1.0 / B1[i];
        A1[i] *= r1; C1[i] *= r1; D1[i] *= r1;
    }
    double* Ajm = A1, *Cjm = C1, *Djm = D1;
    double* Aj = A1 + n_sys, *Bj = B1 + n_sys;
    double* Cj = C1 + n_sys, *Dj = D1 + n_sys;
    // Phase 1: j=2..J — full update (D, C, A)
    for (j = 2; j <= J; ++j,
         Ajm = Aj, Cjm = Cj, Djm = Dj,
         Aj += n_sys, Bj += n_sys, Cj += n_sys, Dj += n_sys) {
        #pragma omp simd
        for (i = 0; i < n_sys; ++i) {
            double inv = 1.0 / (Bj[i] - Aj[i] * Cjm[i]);
            Dj[i] =  inv * (Dj[i] - Aj[i] * Djm[i]);
            Cj[i] =  inv * Cj[i];
            Aj[i] = -inv * Aj[i] * Ajm[i];
        }
    }
    // Phase 2: j=J+1..n_row-1 — skip A update
    for (j = J + 1; j < n_row; ++j,
         Ajm = Aj, Cjm = Cj, Djm = Dj,
         Aj += n_sys, Bj += n_sys, Cj += n_sys, Dj += n_sys) {
        #pragma omp simd
        for (i = 0; i < n_sys; ++i) {
            double inv = 1.0 / (Bj[i] - Aj[i] * Cjm[i]);
            Dj[i] = inv * (Dj[i] - Aj[i] * Djm[i]);
            Cj[i] = inv * Cj[i];
        }
    }
}

// ---------------------------------------------------------------------------
//  Step 2: Backward substitution
// ---------------------------------------------------------------------------

// v1 — single merged pass updating D, A, C for rows n_row-3..1.
[[gnu::always_inline]] static inline
void bwd_subst_v1(double* __restrict A, double* __restrict C,
                  double* __restrict D, int n_sys, int n_row) {
    int i, j;
    double* Ajp = A + (std::size_t)(n_row - 2) * n_sys;
    double* Cjp = C + (std::size_t)(n_row - 2) * n_sys;
    double* Djp = D + (std::size_t)(n_row - 2) * n_sys;
    double* Aj = Ajp - n_sys, *Cj = Cjp - n_sys, *Dj = Djp - n_sys;
    for (j = n_row - 3; j >= 1; --j,
         Ajp = Aj, Cjp = Cj, Djp = Dj,
         Aj -= n_sys, Cj -= n_sys, Dj -= n_sys) {
        #pragma omp simd
        for (i = 0; i < n_sys; ++i) {
            Dj[i] -= Cj[i] * Djp[i];
            Aj[i] -= Cj[i] * Ajp[i];
            Cj[i] *= -Cjp[i];
        }
    }
}

// v2 — 3 separate phases (D all rows; A only rows <J; C only rows >lo).
[[gnu::always_inline]] static inline
void bwd_subst_v2(double* __restrict A, double* __restrict C,
                  double* __restrict D, int n_sys, int n_row,
                  int J, int lo) {
    int i, j;
    {
        double* Cj = C + (std::size_t)(n_row - 3) * n_sys;
        double* Dj = D + (std::size_t)(n_row - 3) * n_sys;
        double* Djp = Dj + n_sys;
        for (j = n_row - 3; j >= 1; --j) {
            #pragma omp simd
            for (i = 0; i < n_sys; ++i) Dj[i] -= Cj[i] * Djp[i];
            Djp = Dj; Dj -= n_sys; Cj -= n_sys;
        }
    }
    if (J >= 2) {
        double* Aj = A + (std::size_t)(J - 1) * n_sys;
        double* Cj = C + (std::size_t)(J - 1) * n_sys;
        double* Ajp = Aj + n_sys;
        for (j = J - 1; j >= 1; --j) {
            #pragma omp simd
            for (i = 0; i < n_sys; ++i) Aj[i] -= Cj[i] * Ajp[i];
            Ajp = Aj; Aj -= n_sys; Cj -= n_sys;
        }
    }
    {
        double* Cjp = C + (std::size_t)(n_row - 2) * n_sys;
        double* Cj  = Cjp - n_sys;
        for (j = n_row - 3; j >= lo + 1; --j) {
            #pragma omp simd
            for (i = 0; i < n_sys; ++i) Cj[i] *= -Cjp[i];
            Cjp = Cj; Cj -= n_sys;
        }
    }
}

// ---------------------------------------------------------------------------
//  Step 3: Pack reduced system — decouple row 0 from x_{N-1}
// ---------------------------------------------------------------------------

[[gnu::always_inline]] static inline
void pack_reduced(double* __restrict A, double* __restrict C,
                  double* __restrict D, int n_sys) {
    double* A1 = A + (std::size_t)1 * n_sys;
    double* D1 = D + (std::size_t)1 * n_sys;
    #pragma omp simd
    for (int i = 0; i < n_sys; ++i) {
        double r = 1.0 / (1.0 - A1[i] * C[i]);
        D[i] = r * (D[i] - C[i] * D1[i]);
        A[i] = r * A[i];
    }
}

// ---------------------------------------------------------------------------
//  Step 5: Solve both interface 2x2 blocks locally (post-exchange)
//    left  : D[0]   = (D0 - A0*D_left)    / (1 - C_left*A0)
//    right : D[N-1] = (DN - CN*D0_right)  / (1 - A_right*CN)
// ---------------------------------------------------------------------------

[[gnu::always_inline]] static inline
void solve_interfaces(double* __restrict A, double* __restrict C,
                                    double* __restrict D,
                                    const double* __restrict C_left,
                                    const double* __restrict D_left,
                                    const double* __restrict A_right,
                                    const double* __restrict D0_right,
                                    bool has_left, bool has_right,
                                    int n_sys, int n_row) {
    if (has_left) {
        #pragma omp simd
        for (int i = 0; i < n_sys; ++i)
            D[i] = (D[i] - A[i] * D_left[i]) / (1.0 - C_left[i] * A[i]);
    }
    if (has_right) {
        double* CN = C + (std::size_t)(n_row - 1) * n_sys;
        double* DN = D + (std::size_t)(n_row - 1) * n_sys;
        #pragma omp simd
        for (int i = 0; i < n_sys; ++i)
            DN[i] = (DN[i] - CN[i] * D0_right[i]) / (1.0 - A_right[i] * CN[i]);
    }
}

// ---------------------------------------------------------------------------
//  Step 6: Final local solve (boundary correction with J cutoff)
//    left  rows 1..left_end    : D[j] -= A[j]*D0
//    right rows right_beg..N-2 : D[j] -= C[j]*DN
//  Callers pass the exact loop bounds (they differ between v1 and v2).
// ---------------------------------------------------------------------------

[[gnu::always_inline]] static inline
void final_solve(double* __restrict A, double* __restrict C,
                 double* __restrict D, int n_sys, int n_row,
                 int left_end, int right_beg) {
    int i, j;
    const double* D0 = D;
    const double* DN = D + (std::size_t)(n_row - 1) * n_sys;
    {
        double* Dj = D + (std::size_t)1 * n_sys;
        double* Aj = A + (std::size_t)1 * n_sys;
        for (j = 1; j <= left_end; ++j, Dj += n_sys, Aj += n_sys) {
            #pragma omp simd
            for (i = 0; i < n_sys; ++i) Dj[i] -= Aj[i] * D0[i];
        }
    }
    {
        double* Dj = D + (std::size_t)right_beg * n_sys;
        double* Cj = C + (std::size_t)right_beg * n_sys;
        for (j = right_beg; j < n_row - 1; ++j, Dj += n_sys, Cj += n_sys) {
            #pragma omp simd
            for (i = 0; i < n_sys; ++i) Dj[i] -= Cj[i] * DN[i];
        }
    }
}

} // namespace ftdma_kernel

#endif // FILTERED_TDMA_KERNELS_HPP
