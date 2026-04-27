// channel/Field.hpp
//
// 3D field with one halo layer in each direction.
//
// Layout matches PaScaL_TCS conventions:
//   - x-fastest (Fortran column-major equivalent)
//   - 0-based index where 0 and n+1 are ghost; 1..n are interior
//   - total storage = (nx+2)*(ny+2)*(nz+2)
//
// Indexing: field(i,j,k) returns data[i + nxt*(j + nyt*k)]

#ifndef CHANNEL_FIELD_HPP
#define CHANNEL_FIELD_HPP

#include <vector>
#include <cstddef>
#include <cstdio>
#include <algorithm>
#include <cassert>

namespace channel {

template <typename T>
class Field {
public:
    Field() = default;

    Field(int nx, int ny, int nz)
        : nx_(nx), ny_(ny), nz_(nz),
          nxt_(nx + 2), nyt_(ny + 2), nzt_(nz + 2),
          data_(static_cast<std::size_t>(nx + 2)
                * static_cast<std::size_t>(ny + 2)
                * static_cast<std::size_t>(nz + 2),
                T{}) {}

    T& operator()(int i, int j, int k) {
        return data_[idx(i, j, k)];
    }
    const T& operator()(int i, int j, int k) const {
        return data_[idx(i, j, k)];
    }

    int nx()  const { return nx_;  }
    int ny()  const { return ny_;  }
    int nz()  const { return nz_;  }
    int nxt() const { return nxt_; }
    int nyt() const { return nyt_; }
    int nzt() const { return nzt_; }

    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }
    std::size_t size() const { return data_.size(); }

    void fill(T v) { std::fill(data_.begin(), data_.end(), v); }

    void swap(Field& other) noexcept {
        std::swap(nx_, other.nx_);
        std::swap(ny_, other.ny_);
        std::swap(nz_, other.nz_);
        std::swap(nxt_, other.nxt_);
        std::swap(nyt_, other.nyt_);
        std::swap(nzt_, other.nzt_);
        data_.swap(other.data_);
    }

private:
    std::size_t idx(int i, int j, int k) const {
        if (!(i >= 0 && i < nxt_) || !(j >= 0 && j < nyt_) || !(k >= 0 && k < nzt_)) {
            std::fprintf(stderr, "[Field::idx OOB] i=%d j=%d k=%d  bounds (nxt=%d nyt=%d nzt=%d)\n",
                         i, j, k, nxt_, nyt_, nzt_);
            std::fflush(stderr);
        }
        assert(i >= 0 && i < nxt_);
        assert(j >= 0 && j < nyt_);
        assert(k >= 0 && k < nzt_);
        return static_cast<std::size_t>(i)
             + static_cast<std::size_t>(nxt_)
               * (static_cast<std::size_t>(j)
                  + static_cast<std::size_t>(nyt_) * static_cast<std::size_t>(k));
    }

    int nx_  = 0, ny_  = 0, nz_  = 0;
    int nxt_ = 0, nyt_ = 0, nzt_ = 0;
    std::vector<T> data_;
};

} // namespace channel

#endif // CHANNEL_FIELD_HPP
