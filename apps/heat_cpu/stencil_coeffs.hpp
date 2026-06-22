#ifndef STENCIL_COEFFS_HPP
#define STENCIL_COEFFS_HPP

/// Stencil coefficients for the 3D ADI scheme (Dirichlet boundary-aware).
struct StencilCoeffs {
    double a, b, c;
};

/// Uniform cell-centre stencil (ghost-cell Dirichlet BC).
inline StencilCoeffs compute_stencil(double dt, double dd) {
    double base = dt / (2.0 * dd * dd);
    return { base, -2.0 * base, base };
}

inline StencilCoeffs compute_stencil_cycl(double dt, double dd, int left_bdy, int right_bdy) {
    double base = dt / (2.0 * dd * dd);
    return {
        base * ( 1.0 ),
        base * (-2.0 ),
        base * ( 1.0 )
    };
}
#endif // STENCIL_COEFFS_HPP
