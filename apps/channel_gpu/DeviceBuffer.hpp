#ifndef CHANNEL_DEVICE_BUFFER_HPP
#define CHANNEL_DEVICE_BUFFER_HPP

#include <cuda_runtime.h>
#include <cstddef>
#include <utility>

#include "GpuUtils.cuh"

namespace channel {

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t n) { reset(n); }
    ~DeviceBuffer() { release(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept { move_from(other); }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            move_from(other);
        }
        return *this;
    }

    void reset(std::size_t n) {
        release();
        n_ = n;
        if (n_ > 0) CHANNEL_CUDA_CHECK(cudaMalloc(&ptr_, n_ * sizeof(T)));
    }

    void release() {
        if (ptr_) cudaFree(ptr_);
        ptr_ = nullptr;
        n_ = 0;
    }

    void zero() {
        if (ptr_ && n_) CHANNEL_CUDA_CHECK(cudaMemset(ptr_, 0, n_ * sizeof(T)));
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    std::size_t size() const { return n_; }

private:
    void move_from(DeviceBuffer& other) noexcept {
        ptr_ = other.ptr_;
        n_ = other.n_;
        other.ptr_ = nullptr;
        other.n_ = 0;
    }

    T* ptr_ = nullptr;
    std::size_t n_ = 0;
};

} // namespace channel

#endif // CHANNEL_DEVICE_BUFFER_HPP
