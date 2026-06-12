#include "tdma_local.hpp"

#include <cstddef>
#include <vector>

void tdma_many(double* __restrict A,
               double* __restrict B,
               double* __restrict C,
               double* __restrict D,
               int n_sys, int n_row) {

    double* B0 = B;
    double* C0 = C;
    double* D0 = D;

    // --- Preprocess (j=0) ---
    #pragma omp simd
    for (int i = 0; i < n_sys; ++i) {
        double inv_b = 1.0 / B0[i];
        D0[i] *= inv_b;
        C0[i] *= inv_b;
    }

    // --- Forward Elimination ---
    for (int j = 1; j < n_row; ++j) {
        double*       Aj  = A + (std::size_t)j * n_sys;
        double*       Bj  = B + (std::size_t)j * n_sys;
        double*       Cj  = C + (std::size_t)j * n_sys;
        double*       Dj  = D + (std::size_t)j * n_sys;
        const double* Cjm = C + (std::size_t)(j - 1) * n_sys;
        const double* Djm = D + (std::size_t)(j - 1) * n_sys;

        #pragma omp simd
        for (int i = 0; i < n_sys; ++i) {
            double r = 1.0 / (Bj[i] - Aj[i] * Cjm[i]);
            Dj[i] = r * (Dj[i] - Aj[i] * Djm[i]);
            Cj[i] = r * Cj[i];
        }
    }

    // --- Backward Substitution ---
    for (int j = n_row - 2; j >= 0; --j) {
        double*       Cj  = C + (std::size_t)j * n_sys;
        double*       Dj  = D + (std::size_t)j * n_sys;
        const double* Djp = D + (std::size_t)(j + 1) * n_sys;

        #pragma omp simd
        for (int i = 0; i < n_sys; ++i) {
            Dj[i] -= Cj[i] * Djp[i];
        }
    }
}

void tdma_single(std::vector<double>& a, std::vector<double>& b,
                 std::vector<double>& c, std::vector<double>& d, int n) {

    d[0] = d[0] / b[0];
    c[0] = c[0] / b[0];

    for (int i = 1; i < n; ++i) {
        double r = 1.0 / (b[i] - a[i] * c[i - 1]);
        d[i] = r * (d[i] - a[i] * d[i - 1]);
        c[i] = r * c[i];
    }

    for (int i = n - 2; i >= 0; --i) {
        d[i] -= c[i] * d[i + 1];
    }
}

void tdma_cyclic_single(std::vector<double>& a, std::vector<double>& b,
                        std::vector<double>& c, std::vector<double>& d, int n) {

    // Sherman-Morrison auxiliary vector e
    std::vector<double> e(n, 0.0);
    e[1]     = -a[1];
    e[n - 1] = -c[n - 1];

    // --- Preprocess (i=1) ---
    d[1] = d[1] / b[1];
    e[1] = e[1] / b[1];
    c[1] = c[1] / b[1];

    // --- Forward Elimination ---
    for (int i = 2; i <= n - 1; ++i) {
        double rr = 1.0 / (b[i] - a[i] * c[i - 1]);
        d[i] = rr * (d[i] - a[i] * d[i - 1]);
        e[i] = rr * (e[i] - a[i] * e[i - 1]);
        c[i] = rr * c[i];
    }

    // --- Backward Substitution ---
    for (int i = n - 2; i >= 1; --i) {
        d[i] -= c[i] * d[i + 1];
        e[i] -= c[i] * e[i + 1];
    }

    // --- Solve for d[0] using cyclic boundary ---
    d[0] = (d[0] - a[0] * d[n - 1] - c[0] * d[1])
         / (b[0] + a[0] * e[n - 1] + c[0] * e[1]);

    // --- Back-substitute d[0] ---
    for (int i = 1; i <= n - 1; ++i) {
        d[i] += d[0] * e[i];
    }
}

void tdma_cyclic_many(double* __restrict A,
                      double* __restrict B,
                      double* __restrict C,
                      double* __restrict D,
                      int n_sys, int n_row) {

    // Auxiliary array E tracks cyclic coupling; same layout [n_row × n_sys]
    std::vector<double> Ev((std::size_t)n_row * n_sys, 0.0);
    double* E = Ev.data();

    double* A1   = A + (std::size_t)1 * n_sys;
    double* CNm1 = C + (std::size_t)(n_row - 1) * n_sys;
    double* E1   = E + (std::size_t)1 * n_sys;
    double* ENm1 = E + (std::size_t)(n_row - 1) * n_sys;

    // Initialize E[1] = -A[1],  E[N-1] = -C[N-1]
    #pragma omp simd
    for (int i = 0; i < n_sys; ++i) {
        E1[i]   = -A1[i];
        ENm1[i] = -CNm1[i];
    }

    // --- Preprocess row 1 ---
    {
        double* B1 = B + (std::size_t)1 * n_sys;
        double* C1 = C + (std::size_t)1 * n_sys;
        double* D1 = D + (std::size_t)1 * n_sys;
        #pragma omp simd
        for (int i = 0; i < n_sys; ++i) {
            double inv_b = 1.0 / B1[i];
            D1[i] *= inv_b;
            E1[i] *= inv_b;
            C1[i] *= inv_b;
        }
    }

    // --- Forward Elimination rows 2..N-1 ---
    for (int j = 2; j < n_row; ++j) {
        double*       Aj  = A + (std::size_t)j * n_sys;
        double*       Bj  = B + (std::size_t)j * n_sys;
        double*       Cj  = C + (std::size_t)j * n_sys;
        double*       Dj  = D + (std::size_t)j * n_sys;
        double*       Ej  = E + (std::size_t)j * n_sys;
        const double* Cjm = C + (std::size_t)(j - 1) * n_sys;
        const double* Djm = D + (std::size_t)(j - 1) * n_sys;
        const double* Ejm = E + (std::size_t)(j - 1) * n_sys;

        #pragma omp simd
        for (int i = 0; i < n_sys; ++i) {
            double r = 1.0 / (Bj[i] - Aj[i] * Cjm[i]);
            Dj[i] = r * (Dj[i] - Aj[i] * Djm[i]);
            Ej[i] = r * (Ej[i] - Aj[i] * Ejm[i]);
            Cj[i] = r * Cj[i];
        }
    }

    // --- Backward Substitution rows N-2..1 ---
    for (int j = n_row - 2; j >= 1; --j) {
        double*       Cj  = C + (std::size_t)j * n_sys;
        double*       Dj  = D + (std::size_t)j * n_sys;
        double*       Ej  = E + (std::size_t)j * n_sys;
        const double* Djp = D + (std::size_t)(j + 1) * n_sys;
        const double* Ejp = E + (std::size_t)(j + 1) * n_sys;

        #pragma omp simd
        for (int i = 0; i < n_sys; ++i) {
            Dj[i] -= Cj[i] * Djp[i];
            Ej[i] -= Cj[i] * Ejp[i];
        }
    }

    // --- Solve for D[0] using cyclic boundary ---
    //   A[0]*x[N-1] + B[0]*x[0] + C[0]*x[1] = D[0]
    //   x[j] = D[j] + x[0]*E[j]  for j=1..N-1
    //   → x[0] = (D[0] - A[0]*D[N-1] - C[0]*D[1])
    //           / (B[0] + A[0]*E[N-1] + C[0]*E[1])
    {
        double*       D0   = D;
        const double* A0   = A;
        const double* B0   = B;
        const double* C0   = C;
        const double* D1   = D + (std::size_t)1 * n_sys;
        const double* DNm1 = D + (std::size_t)(n_row - 1) * n_sys;

        #pragma omp simd
        for (int i = 0; i < n_sys; ++i) {
            D0[i] = (D0[i] - A0[i] * DNm1[i] - C0[i] * D1[i])
                  / (B0[i] + A0[i] * ENm1[i] + C0[i] * E1[i]);
        }
    }

    // --- Back-substitute D[0] into rows 1..N-1 ---
    for (int j = 1; j < n_row; ++j) {
        double*       Dj = D + (std::size_t)j * n_sys;
        const double* Ej = E + (std::size_t)j * n_sys;
        const double* D0 = D;
        #pragma omp simd
        for (int i = 0; i < n_sys; ++i)
            Dj[i] += D0[i] * Ej[i];
    }
}
