// Milestone 3: DeviceBuffer<T> and GpuTimer
//
// Practices: RAII, destructors, deleting copies, move constructor and move
// assignment, noexcept, explicit, templates.
//
// Fill in every TODO 3.x, then run `make gpu` on a machine with an NVIDIA GPU.

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Given: wrap every CUDA API call in this. It prints the error and exits.
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error \"%s\" at %s:%d\n",                \
                         cudaGetErrorString(err_), __FILE__, __LINE__);         \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// Owns an array of T in GPU memory. Frees it automatically.
template <typename T>
class DeviceBuffer
{
private:
    T*          m_ptr{ nullptr };
    std::size_t m_size{ 0 }; // number of elements, not bytes

public:
    DeviceBuffer() = default;

    // TODO 3.1: allocate n elements with cudaMalloc (inside CUDA_CHECK).
    // Why is this constructor explicit? (lesson 14.16)
    explicit DeviceBuffer(std::size_t n)
    {
    }

    // TODO 3.2: free the memory. cudaFree(nullptr) is safe.
    ~DeviceBuffer()
    {
    }

    // TODO 3.3: delete the copy constructor and copy assignment operator.
    // What would happen if two DeviceBuffers held the same pointer and both
    // destructors ran?

    // TODO 3.4: move constructor. Take other's pointer and size, then leave
    // other empty (nullptr, 0) so its destructor does nothing.
    DeviceBuffer(DeviceBuffer&& other) noexcept
    {
    }

    // TODO 3.5: move assignment. Guard against self-assignment, free the memory
    // you currently own, take other's, leave other empty, return *this.
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        return *this;
    }

    // TODO 3.6: accessors.
    T* data() { return nullptr; }
    const T* data() const { return nullptr; }
    std::size_t size() const { return 0; }

    // TODO 3.7: copy host -> device with cudaMemcpy. Assume host.size() == size().
    void upload(const std::vector<T>& host)
    {
    }

    // TODO 3.8: copy device -> host. Resize host to size() first.
    void download(std::vector<T>& host) const
    {
    }
};

// Times GPU work with CUDA events. Also RAII: events are created in the
// constructor and destroyed in the destructor.
class GpuTimer
{
private:
    cudaEvent_t m_start{};
    cudaEvent_t m_stop{};

public:
    // TODO 3.9: create both events (cudaEventCreate).
    GpuTimer()
    {
    }

    // TODO 3.10: destroy both events (cudaEventDestroy).
    ~GpuTimer()
    {
    }

    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    // TODO 3.11: record m_start (cudaEventRecord).
    void start()
    {
    }

    // TODO 3.12: record m_stop, then wait for it (cudaEventSynchronize).
    void stop()
    {
    }

    // TODO 3.13: milliseconds between start and stop (cudaEventElapsedTime).
    float elapsedMs() const
    {
        return 0.0f;
    }
};
