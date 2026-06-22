#ifndef STENCIL_COEFFS_HPP
#define STENCIL_COEFFS_HPP

/// Stencil coefficients for the 3D ADI scheme (Dirichlet boundary-aware).
struct StencilCoeffs {
    double a, b, c;
};

/// Uniform cell-centre stencil (ghost-cell Dirichlet BC).
/// All rows identical → normalized off-diagonal ρ = rho on every row (< 1/2).
inline StencilCoeffs compute_stencil(double dt, double dd) {
    double base = dt / (2.0 * dd * dd);
    return { base, -2.0 * base, base };
}

#endif // STENCIL_COEFFS_HPP
