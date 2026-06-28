#include "tdma_local.hpp"
#include <vector>

void ftdma_many(double* __restrict A,
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
        double* Aj  = A + (std::size_t)j * n_sys;
        double* Bj  = B + (std::size_t)j * n_sys;
        double* Cj  = C + (std::size_t)j * n_sys;
        double* Dj  = D + (std::size_t)j * n_sys;
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
        double* Cj  = C + (std::size_t)j * n_sys;
        double* Dj  = D + (std::size_t)j * n_sys;
        const double* Djp = D + (std::size_t)(j + 1) * n_sys;

        #pragma omp simd
        for (int i = 0; i < n_sys; ++i) {
            Dj[i] -= Cj[i] * Djp[i];
        }
    }
}

void ftdma_cyclic_many(double* __restrict A,
                    double* __restrict B,
                    double* __restrict C,
                    double* __restrict D,
                    int n_sys,
                    int n_row) {

    std::vector<double> e((std::size_t)n_sys * n_row, 0.0);

    for (int i = 0; i < n_sys; ++i) {
        e[(std::size_t)1 * n_sys + i]           = -A[(std::size_t)1 * n_sys + i];
        e[(std::size_t)(n_row - 1) * n_sys + i] = -C[(std::size_t)(n_row - 1) * n_sys + i];
    }

    {
        double* B1 = B + (std::size_t)1 * n_sys;
        double* C1 = C + (std::size_t)1 * n_sys;
        double* D1 = D + (std::size_t)1 * n_sys;
        double* E1 = e.data() + (std::size_t)1 * n_sys;

        #pragma omp simd
        for (int i = 0; i < n_sys; ++i) {
            double inv_b = 1.0 / B1[i];
            D1[i] *= inv_b;
            E1[i] *= inv_b;
            C1[i] *= inv_b;
        }
    }

    for (int j = 2; j < n_row; ++j) {
        double* Aj  = A + (std::size_t)j * n_sys;
        double* Bj  = B + (std::size_t)j * n_sys;
        double* Cj  = C + (std::size_t)j * n_sys;
        double* Dj  = D + (std::size_t)j * n_sys;
        double* Ej  = e.data() + (std::size_t)j * n_sys;

        const double* Cjm = C + (std::size_t)(j - 1) * n_sys;
        const double* Djm = D + (std::size_t)(j - 1) * n_sys;
        const double* Ejm = e.data() + (std::size_t)(j - 1) * n_sys;

        #pragma omp simd
        for (int i = 0; i < n_sys; ++i) {
            double r = 1.0 / (Bj[i] - Aj[i] * Cjm[i]);
            Dj[i] = r * (Dj[i] - Aj[i] * Djm[i]);
            Ej[i] = r * (Ej[i] - Aj[i] * Ejm[i]);
            Cj[i] = r * Cj[i];
        }
    }

    for (int j = n_row - 2; j >= 1; --j) {
        double* Cj  = C + (std::size_t)j * n_sys;
        double* Dj  = D + (std::size_t)j * n_sys;
        double* Ej  = e.data() + (std::size_t)j * n_sys;

        const double* Djp = D + (std::size_t)(j + 1) * n_sys;
        const double* Ejp = e.data() + (std::size_t)(j + 1) * n_sys;

        #pragma omp simd
        for (int i = 0; i < n_sys; ++i) {
            Dj[i] -= Cj[i] * Djp[i];
            Ej[i] -= Cj[i] * Ejp[i];
        }
    }

    {
        double* A0 = A + (std::size_t)0 * n_sys;
        double* B0 = B + (std::size_t)0 * n_sys;
        double* C0 = C + (std::size_t)0 * n_sys;
        double* D0 = D + (std::size_t)0 * n_sys;

        const double* D1 = D + (std::size_t)1 * n_sys;
        const double* DN = D + (std::size_t)(n_row - 1) * n_sys;
        const double* E1 = e.data() + (std::size_t)1 * n_sys;
        const double* EN = e.data() + (std::size_t)(n_row - 1) * n_sys;

        #pragma omp simd
        for (int i = 0; i < n_sys; ++i) {
            D0[i] = (D0[i] - A0[i] * DN[i] - C0[i] * D1[i])
                  / (B0[i] + A0[i] * EN[i] + C0[i] * E1[i]);
        }
    }

    for (int j = 1; j < n_row; ++j) {
        double* Dj = D + (std::size_t)j * n_sys;
        double* Ej = e.data() + (std::size_t)j * n_sys;
        const double* D0 = D + (std::size_t)0 * n_sys;

        #pragma omp simd
        for (int i = 0; i < n_sys; ++i) {
            Dj[i] += D0[i] * Ej[i];
        }
    }
}