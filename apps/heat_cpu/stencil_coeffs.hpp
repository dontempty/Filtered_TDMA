#ifndef STENCIL_COEFFS_HPP
#define STENCIL_COEFFS_HPP

/// Stencil coefficients for the 3D ADI scheme (Dirichlet boundary-aware).
struct StencilCoeffs {
    double a, b, c;
};

/// Compute stencil coefficients for a given direction.
///   dt       : time step
///   dd       : grid spacing in this direction
///   left_bdy : 1 if this cell is adjacent to the left physical boundary, else 0
///   right_bdy: 1 if this cell is adjacent to the right physical boundary, else 0
inline StencilCoeffs compute_stencil(double dt, double dd, int left_bdy, int right_bdy) {
    double base = dt / (2.0 * dd * dd);
    return {
        base * ( 1.0 + (5.0/3.0) * left_bdy + (1.0/3.0) * right_bdy),
        base * (-2.0 -     (2.0) * left_bdy  -     (2.0) * right_bdy),
        base * ( 1.0 + (1.0/3.0) * left_bdy  + (5.0/3.0) * right_bdy)
    };
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
