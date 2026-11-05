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
    T*          m_ptr{ nullptr }; // &m_ptr cpu addr, m_ptr gpu addr, m_ptr[0] gpu addr
    std::size_t m_size{ 0 }; // number of elements, not bytes

public:
    DeviceBuffer() = default;

    // TODO 3.1: allocate n elements with cudaMalloc (inside CUDA_CHECK).
    // Why is this constructor explicit? (lesson 14.16)
    // exp
    explicit DeviceBuffer(std::size_t n)
        : m_size{n}
    {
        std::size_t n_bytes{n * sizeof(T)}; //too big for int
        CUDA_CHECK(cudaMalloc(&m_ptr, n_bytes)); //outputs thing to &m_ptr

        // needs &m_ptr bc it overwrites the address stored in m_ptr
        // errors are return codes, show up later, macro API call

        //m_ptr on cpu pts to 

//         cudaError_t cudaMalloc(void** devPtr, size_t size);
// //                     ^^^^^^ pointer to a pointer
    }

    // TODO 3.2: free the memory. cudaFree(nullptr) is safe.
    ~DeviceBuffer()
    {
        CUDA_CHECK(cudaFree(m_ptr));
    }

    // TODO 3.3: delete the copy constructor and copy assignment operator.
    // What would happen if two DeviceBuffers held the same pointer and both
    // destructors ran?
    DeviceBuffer(const DeviceBuffer&) = delete; //param is source
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // TODO 3.4: move constructor. Take other's pointer and size, then leave
    // other empty (nullptr, 0) so its destructor does nothing.
    DeviceBuffer(DeviceBuffer&& other) noexcept
    : m_ptr{other.m_ptr}, m_size{other.m_size} //private per class not object
    {
        other.m_ptr = nullptr;
        other.m_size = 0;
    }
    // DeviceBuffer b = std::move(a); B IS NEW

    // TODO 3.5: move assignment. Guard against self-assignment, free the memory
    // you currently own, take other's, leave other empty, return *this.
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other) { //this already ptr. other is a reference
            CUDA_CHECK(cudaFree(m_ptr)); // free the gpu memory
            m_ptr = other.m_ptr; m_size = other.m_size;
            other.m_ptr = nullptr;
            other.m_size = 0;
        }

        return *this;
    }
    // b = std::move(a); B IS EXISTS

    // TODO 3.6: accessors.
    T* data() { return m_ptr; } // gpu addr
    const T* data() const { return m_ptr; } //read only
    std::size_t size() const { return m_size; }

    // TODO 3.7: copy host -> device with cudaMemcpy. Assume host.size() == size().
    void upload(const std::vector<T>& host)
    {
        //c arrays decay to pointer std::vector class so no decay. 
        // host is an entire object, host.data() is pointer to actual data
        CUDA_CHECK(cudaMemcpy(m_ptr, host.data(), sizeof(T)*host.size(), cudaMemcpyHostToDevice));
    }

    // TODO 3.8: copy device -> host. Resize host to size() first.
    void download(std::vector<T>& host) const
    {
        //writign into host not const
        host.resize(m_size);
        CUDA_CHECK(cudaMemcpy(host.data(), m_ptr, sizeof(T)*host.size(), cudaMemcpyDeviceToHost));
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
