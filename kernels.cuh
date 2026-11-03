// Milestones 4, 5 and 6 (GPU half): the kernels
//
// Each kernel has a launcher below it. The launcher picks the grid and block
// sizes and launches the kernel; the test harness calls the launchers.
//
// Fill in every TODO, then run `make gpu`.

#pragma once

#include "cpu_analysis.h"
#include "device_buffer.cuh"

constexpr int BLOCK_SIZE = 256; // threads per block (a multiple of the warp size, 32)

// Given: a sensible grid size for n elements. Enough blocks to fill every
// SM several times over, without launching millions of tiny blocks; the
// kernels cover any leftover elements with a grid-stride loop.
inline int gridSizeFor(int n)
{
    int device = 0;
    int numSMs = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));
    const int needed = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int cap = numSMs * 8;
    return needed < cap ? needed : cap;
}

// ============================================================================
// Milestone 4: per-triangle area, SoA vs AoS
// ============================================================================

// TODO 4.1: one triangle per thread: out[t] = area of triangle t.
// Use a grid-stride loop: start at this thread's global index and step by the
// total number of threads in the grid, so any n works with any grid size.
// Call triangleArea from mesh.h (it's HOST_DEVICE, so it runs on the GPU).
template <typename Real>
__global__ void areaKernelSoA(const Real* x0, const Real* y0, const Real* x1, const Real* y1,
                              const Real* x2, const Real* y2, Real* out, int n)
{
}

// TODO 4.2: the same, reading from an array of TriCoords structs.
template <typename Real>
__global__ void areaKernelAoS(const TriCoords<Real>* tris, Real* out, int n)
{
}

// TODO 4.3: launch areaKernelSoA with gridSizeFor(n) blocks of BLOCK_SIZE threads.
template <typename Real>
void launchAreaSoA(const Real* x0, const Real* y0, const Real* x1, const Real* y1, const Real* x2,
                   const Real* y2, Real* out, int n)
{
}

// TODO 4.4: launch areaKernelAoS.
template <typename Real>
void launchAreaAoS(const TriCoords<Real>* tris, Real* out, int n)
{
}

// ============================================================================
// Milestone 5: sum an array (parallel reduction), three ways
// ============================================================================
// In all three, *out has been zeroed by the caller before the launch.

// TODO 5.1: naive. Every thread atomicAdds each of its elements into *out.
__global__ void reduceAtomicKernel(const float* in, float* out, int n)
{
}

inline void launchReduceAtomic(const float* in, float* out, int n)
{
    // TODO 5.1: launch it
}

// TODO 5.2: shared-memory tree reduction.
//   1. Each thread sums its grid-stride elements into a local variable.
//   2. It writes that to a __shared__ float array of BLOCK entries.
//      __syncthreads().
//   3. Tree step: for s = BLOCK/2, BLOCK/4, ..., 1, threads with
//      threadIdx.x < s add element [threadIdx.x + s] into [threadIdx.x].
//      __syncthreads() after every step. (Why does every thread have to reach
//      the __syncthreads(), even the ones that did no work?)
//   4. Thread 0 atomicAdds the block's total into *out.
// BLOCK is a template parameter, so the shared array size is a compile-time
// constant.
template <int BLOCK>
__global__ void reduceSharedKernel(const float* in, float* out, int n)
{
}

inline void launchReduceShared(const float* in, float* out, int n)
{
    // TODO 5.2: launch reduceSharedKernel<BLOCK_SIZE>
}

// TODO 5.3: warp-shuffle reduction.
//   1. Same per-thread grid-stride sum.
//   2. Reduce within each warp with __shfl_down_sync(0xffffffff, v, offset)
//      for offset = 16, 8, 4, 2, 1. Afterwards lane 0 holds the warp's sum.
//   3. Lane 0 of each warp writes its sum to __shared__ float warpSums[BLOCK / 32].
//      __syncthreads().
//   4. The first warp loads warpSums (0 for lanes past BLOCK / 32), reduces
//      them with shuffles again, and thread 0 atomicAdds into *out.
template <int BLOCK>
__global__ void reduceShuffleKernel(const float* in, float* out, int n)
{
}

inline void launchReduceShuffle(const float* in, float* out, int n)
{
    // TODO 5.3: launch reduceShuffleKernel<BLOCK_SIZE>
}

// ============================================================================
// Milestone 6 (GPU half): sparse matrix-vector product, y = A * x
// ============================================================================

// TODO 6.3: one row per thread (use a grid-stride loop over rows):
//   y[row] = sum over k in [rowPtr[row], rowPtr[row + 1]) of values[k] * x[colIdx[k]]
template <typename Real>
__global__ void spmvKernel(int rows, const int* rowPtr, const int* colIdx, const Real* values,
                           const Real* x, Real* y)
{
}

// TODO 6.4: launch spmvKernel over `rows` rows.
template <typename Real>
void launchSpmv(int rows, const int* rowPtr, const int* colIdx, const Real* values, const Real* x,
                Real* y)
{
}
