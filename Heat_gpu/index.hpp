#ifndef INDEX_UTIL_HPP
#define INDEX_UTIL_HPP

// ---- 3D linearization ----

inline int idx_ijk(int i, int j, int k, int nx, int ny) {
    return k * ny * nx + j * nx + i;
}

// ---- 2D linearization (for boundary slabs) ----

inline int idx_ij(int i, int j, int nx) { return j * nx + i; }
inline int idx_ik(int i, int k, int nx) { return k * nx + i; }
inline int idx_ji(int j, int i, int ny) { return i * ny + j; }
inline int idx_jk(int j, int k, int ny) { return k * ny + j; }

#endif // INDEX_UTIL_HPP
