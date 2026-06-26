#ifndef CHANNEL_DEVICE_FIELD_HPP
#define CHANNEL_DEVICE_FIELD_HPP

#include <cuda_runtime.h>
#include <cstddef>
#include <utility>

#include "Field.hpp"
#include "GpuUtils.cuh"

namespace channel {

class DeviceField {
public:
    DeviceField() = default;
    DeviceField(int nx, int ny, int nz) { reset(nx, ny, nz); }
    ~DeviceField() { release(); }

    DeviceField(const DeviceField&) = delete;
    DeviceField& operator=(const DeviceField&) = delete;

    DeviceField(DeviceField&& other) noexcept { move_from(other); }
    DeviceField& operator=(DeviceField&& other) noexcept {
        if (this != &other) {
            release();
            move_from(other);
        }
        return *this;
    }

    void reset(int nx, int ny, int nz) {
        release();
        nx_ = nx; ny_ = ny; nz_ = nz;
        nxt_ = nx + 2; nyt_ = ny + 2; nzt_ = nz + 2;
        n_ = static_cast<std::size_t>(nxt_) * nyt_ * nzt_;
        CHANNEL_CUDA_CHECK(cudaMalloc(&ptr_, n_ * sizeof(double)));
        CHANNEL_CUDA_CHECK(cudaMemset(ptr_, 0, n_ * sizeof(double)));
    }

    void release() {
        if (ptr_) cudaFree(ptr_);
        ptr_ = nullptr;
        nx_ = ny_ = nz_ = nxt_ = nyt_ = nzt_ = 0;
        n_ = 0;
    }

    void fill(double v);

    void copy_from_host(const Field<double>& h) {
        if (h.nx() != nx_ || h.ny() != ny_ || h.nz() != nz_) reset(h.nx(), h.ny(), h.nz());
        CHANNEL_CUDA_CHECK(cudaMemcpy(ptr_, h.data(), n_ * sizeof(double),
                                      cudaMemcpyHostToDevice));
    }

    void copy_to_host(Field<double>& h) const {
        if (h.nx() != nx_ || h.ny() != ny_ || h.nz() != nz_) h = Field<double>(nx_, ny_, nz_);
        CHANNEL_CUDA_CHECK(cudaMemcpy(h.data(), ptr_, n_ * sizeof(double),
                                      cudaMemcpyDeviceToHost));
    }

    void swap(DeviceField& other) noexcept {
        std::swap(nx_, other.nx_);   std::swap(ny_, other.ny_);
        std::swap(nz_, other.nz_);   std::swap(nxt_, other.nxt_);
        std::swap(nyt_, other.nyt_); std::swap(nzt_, other.nzt_);
        std::swap(n_, other.n_);     std::swap(ptr_, other.ptr_);
    }

    double* data() { return ptr_; }
    const double* data() const { return ptr_; }
    std::size_t size() const { return n_; }

    int nx() const { return nx_; }
    int ny() const { return ny_; }
    int nz() const { return nz_; }
    int nxt() const { return nxt_; }
    int nyt() const { return nyt_; }
    int nzt() const { return nzt_; }

private:
    void move_from(DeviceField& other) noexcept {
        nx_ = other.nx_; ny_ = other.ny_; nz_ = other.nz_;
        nxt_ = other.nxt_; nyt_ = other.nyt_; nzt_ = other.nzt_;
        n_ = other.n_; ptr_ = other.ptr_;
        other.nx_ = other.ny_ = other.nz_ = 0;
        other.nxt_ = other.nyt_ = other.nzt_ = 0;
        other.n_ = 0; other.ptr_ = nullptr;
    }

    int nx_ = 0, ny_ = 0, nz_ = 0;
    int nxt_ = 0, nyt_ = 0, nzt_ = 0;
    std::size_t n_ = 0;
    double* ptr_ = nullptr;
};

inline __host__ __device__ std::size_t df_idx(int i, int j, int k, int nxt, int nyt)
{
    return static_cast<std::size_t>(i)
         + static_cast<std::size_t>(nxt)
         * (static_cast<std::size_t>(j)
         + static_cast<std::size_t>(nyt) * static_cast<std::size_t>(k));
}

} // namespace channel

#endif // CHANNEL_DEVICE_FIELD_HPP
