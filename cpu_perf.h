// Milestone 8 (optional): CPU performance
//
// Practices: dependency chains and multiple accumulators, auto-vectorization
// (SIMD), __restrict__, AoS vs SoA on the CPU, std::thread, false sharing,
// cache blocking.
//
// Fill in every TODO 8.x, then run `make perf`. The harness checks each
// function and times it; the timings are the point, so read them and answer
// the questions at the bottom of this file.
//
// `make vec` prints which loops the compiler vectorized (and why not).

#pragma once

#include "mesh.h"

#include <atomic>
#include <thread>
#include <vector>

// ============================================================================
// 8.1-8.2: summing an array, one accumulator vs four
// ============================================================================
// Floating-point addition isn't associative, so without -ffast-math the
// compiler must add in exactly the order you wrote. One accumulator means
// every add waits for the previous one (a dependency chain).
// Heads-up: `make vec` may call these loops "vectorized" anyway. On ARM, clang
// does vector loads but still adds the lanes one by one, in order (a "strict"
// reduction), so the chain remains. Trust the timings.

// TODO 8.1: plain loop, one float accumulator.
inline float sumSimple(const float* a, int n)
{
    return 0.0f;
}

// TODO 8.2: four independent float accumulators (elements i, i+1, i+2, i+3
// go to s0..s3), combined at the end. Handle the n % 4 leftover elements.
inline float sumFourAcc(const float* a, int n)
{
    return 0.0f;
}

// ============================================================================
// 8.3-8.4: per-triangle area, SoA vs AoS, on the CPU
// ============================================================================
// Same computation as Milestone 4, but here the question is whether the
// compiler can turn the loop into SIMD instructions (4 floats at a time on
// ARM NEON, 8 on AVX2).

// Given: one triangle's corners, packed together (AoS).
struct TriF
{
    float x0, y0, x1, y1, x2, y2;
};

// TODO 8.3: out[t] = triangleArea(...) for t in [0, n), reading six separate
// arrays. __restrict__ promises the compiler the arrays don't overlap. Keep
// the loop body simple so it vectorizes; check with `make vec`.
inline void areasSoA(const float* __restrict__ x0, const float* __restrict__ y0,
                     const float* __restrict__ x1, const float* __restrict__ y1,
                     const float* __restrict__ x2, const float* __restrict__ y2,
                     float* __restrict__ out, int n)
{
}

// TODO 8.4: the same, reading an array of TriF structs.
inline void areasAoS(const TriF* __restrict__ tris, float* __restrict__ out, int n)
{
}

// ============================================================================
// 8.5: multithreaded sum with std::thread
// ============================================================================
// std::thread t{ f, arg1, arg2 };  starts f(arg1, arg2) on a new thread.
// t.join();                         waits for it to finish. Every thread must
//                                   be joined before it's destroyed.
// A lambda is the usual f:  std::thread t{ [&] { ... } };

// TODO 8.5: split [0, n) into numThreads contiguous chunks (the last chunk
// takes the remainder). Each thread sums its chunk into a LOCAL double, then
// writes it to its own slot of a partial-sums vector. Join all threads, then
// add up the partials.
inline double sumThreads(const float* a, int n, int numThreads)
{
    return 0.0;
}

// ============================================================================
// 8.6: false sharing
// ============================================================================
// The CPU keeps caches coherent per cache line (64 bytes on x86, 128 on Apple
// silicon). If two cores write different variables that share a line, the
// line bounces between their caches on every write, even though the threads
// never touch each other's data.

// Given: two counter types. Counter is 8 bytes, so many fit in one cache line;
// PaddedCounter is aligned (and so padded) to 128 bytes, one per line.
struct Counter
{
    std::atomic<long> v{ 0 };
};
struct alignas(128) PaddedCounter
{
    std::atomic<long> v{ 0 };
};

// TODO 8.6: start numThreads threads; thread t calls
//   counters[t].v.fetch_add(1, std::memory_order_relaxed)
// iters times. Join them all. (C is Counter or PaddedCounter; the code is the
// same for both, only the memory layout differs.)
template <typename C>
void countParallel(C* counters, int numThreads, long iters)
{
}

// ============================================================================
// 8.7: cache blocking, matrix transpose
// ============================================================================
// in and out are n x n, row-major: element (row r, col c) is at [r * n + c].

// TODO 8.7a: out[c * n + r] = in[r * n + c], two plain nested loops.
inline void transposeNaive(const float* __restrict__ in, float* __restrict__ out, int n)
{
}

// TODO 8.7b: the same result, but work in B x B tiles: the outer two loops
// step over tiles (by B), the inner two loops transpose one tile. Assume n is
// a multiple of B. Why does this help? Think about which of in/out the naive
// version reads or writes with a stride of n floats.
template <int B>
void transposeBlocked(const float* __restrict__ in, float* __restrict__ out, int n)
{
}

// ============================================================================
// Questions (answer in RESULTS.md once your timings are in)
// ============================================================================
// Q8.1  How much faster is sumFourAcc than sumSimple? Why isn't the answer
//       "it does the same number of adds, so it's the same speed"?
// Q8.2  Do sumSimple and sumFourAcc return exactly the same float? Why not?
// Q8.3  Did `make vec` report both area loops vectorized? Which ran faster,
//       and how does that compare with what you'd expect on a GPU?
// Q8.4  How does sumThreads scale from 1 to 8 threads? What limits it?
// Q8.5  How much slower is Counter than PaddedCounter? Explain with cache lines.
// Q8.6  How much faster is the blocked transpose? Which access was the
//       problem in the naive one, and how does the tile fix it?
// Q8.7  In Milestone 2.3, a sort comparator that computes areas does about
//       2 log n area computations per triangle. How would you fix that?
