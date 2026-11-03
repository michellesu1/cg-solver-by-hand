// Tests and timings for Milestone 8 (CPU performance). Runs on your Mac.
//   make perf
// Don't edit this file.

#include "cpu_perf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

namespace {

int g_failures = 0;
volatile double g_sink = 0.0; // stops the compiler from deleting timed work

void check(bool ok, const char* name)
{
    std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok)
        ++g_failures;
}

// Best of `reps` runs, in milliseconds.
template <typename F>
double timeMs(F&& f, int reps = 5)
{
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        const auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return best;
}

void row(const char* name, double ms, double bytes)
{
    std::printf("    %-34s %9.3f ms  %8.2f GB/s\n", name, ms, bytes / (ms * 1e6));
}

// Deterministic pseudo-random floats in [0, 1).
std::vector<float> randomFloats(std::size_t n, unsigned seed)
{
    std::vector<float> v(n);
    unsigned s = seed;
    for (auto& x : v) {
        s = s * 1664525u + 1013904223u;
        x = static_cast<float>(s >> 8) / 16777216.0f;
    }
    return v;
}

double referenceSum(const std::vector<float>& v)
{
    double s = 0.0;
    for (float x : v)
        s += x;
    return s;
}

void testSums()
{
    std::puts("== 8.1-8.2: one accumulator vs four");

    const std::vector<float> small{ 1, 2, 3, 4, 5, 6, 7 }; // 7: exercises the n % 4 tail
    check(sumSimple(small.data(), 7) == 28.0f, "sumSimple small");
    check(sumFourAcc(small.data(), 7) == 28.0f, "sumFourAcc small (with leftover elements)");
    check(sumFourAcc(small.data(), 0) == 0.0f, "sumFourAcc of nothing is 0");

    // 64K floats = 256 KB: fits in cache, so the timing shows the adds, not memory.
    const int n = 1 << 16;
    const auto a = randomFloats(n, 1);
    const double ref = referenceSum(a);
    const float s1 = sumSimple(a.data(), n);
    const float s4 = sumFourAcc(a.data(), n);
    check(std::abs(s1 - ref) / ref < 1e-4 && std::abs(s4 - ref) / ref < 1e-4, "sums match a double reference");
    std::printf("    sumSimple = %.6f, sumFourAcc = %.6f, double ref = %.6f\n", static_cast<double>(s1),
                static_cast<double>(s4), ref);

    const int reps = 200;
    const double bytes = static_cast<double>(reps) * n * sizeof(float);
    row("sumSimple   (x200)", timeMs([&] { for (int r = 0; r < reps; ++r) g_sink = g_sink + sumSimple(a.data(), n); }), bytes);
    row("sumFourAcc  (x200)", timeMs([&] { for (int r = 0; r < reps; ++r) g_sink = g_sink + sumFourAcc(a.data(), n); }), bytes);
}

void testAreas()
{
    std::puts("\n== 8.3-8.4: per-triangle area, SoA vs AoS");

    const int n = 1 << 22; // 4M triangles
    std::vector<float> x0 = randomFloats(n, 2), y0 = randomFloats(n, 3), x1 = randomFloats(n, 4),
                       y1 = randomFloats(n, 5), x2 = randomFloats(n, 6), y2 = randomFloats(n, 7);
    std::vector<TriF> aos(n);
    for (int t = 0; t < n; ++t)
        aos[t] = TriF{ x0[t], y0[t], x1[t], y1[t], x2[t], y2[t] };

    std::vector<float> outSoA(n, -1.0f), outAoS(n, -1.0f);
    areasSoA(x0.data(), y0.data(), x1.data(), y1.data(), x2.data(), y2.data(), outSoA.data(), n);
    areasAoS(aos.data(), outAoS.data(), n);

    bool okSoA = true, okAoS = true;
    for (int t = 0; t < n; t += 997) {
        const float e = triangleArea(x0[t], y0[t], x1[t], y1[t], x2[t], y2[t]);
        okSoA = okSoA && std::abs(outSoA[t] - e) <= 1e-6f;
        okAoS = okAoS && std::abs(outAoS[t] - e) <= 1e-6f;
    }
    okSoA = okSoA && std::abs(outSoA[n - 1] - triangleArea(x0[n - 1], y0[n - 1], x1[n - 1], y1[n - 1], x2[n - 1], y2[n - 1])) <= 1e-6f;
    check(okSoA, "areasSoA");
    check(okAoS, "areasAoS");

    const double bytes = static_cast<double>(n) * 7 * sizeof(float); // 6 reads + 1 write per triangle
    row("areasSoA", timeMs([&] { areasSoA(x0.data(), y0.data(), x1.data(), y1.data(), x2.data(), y2.data(), outSoA.data(), n); }), bytes);
    row("areasAoS", timeMs([&] { areasAoS(aos.data(), outAoS.data(), n); }), bytes);
}

void testThreads()
{
    std::puts("\n== 8.5: multithreaded sum");

    const std::vector<float> small{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    check(sumThreads(small.data(), 10, 3) == 55.0, "sumThreads small, uneven chunks");
    check(sumThreads(small.data(), 10, 1) == 55.0, "sumThreads with 1 thread");

    const int n = 1 << 25; // 32M floats = 128 MB: bigger than every cache
    const auto a = randomFloats(n, 8);
    const double ref = referenceSum(a);
    check(std::abs(sumThreads(a.data(), n, 8) - ref) / ref < 1e-9, "sumThreads matches reference");

    const double bytes = static_cast<double>(n) * sizeof(float);
    const unsigned hw = std::thread::hardware_concurrency();
    std::printf("    (this machine reports %u hardware threads)\n", hw);
    for (int t : { 1, 2, 4, 8 }) {
        char name[64];
        std::snprintf(name, sizeof name, "sumThreads, %d thread%s", t, t == 1 ? "" : "s");
        row(name, timeMs([&] { g_sink = g_sink + sumThreads(a.data(), n, t); }), bytes);
    }
}

void testFalseSharing()
{
    std::puts("\n== 8.6: false sharing");

    const int threads = 4;
    const long iters = 20'000'000;

    auto shared = std::make_unique<Counter[]>(threads);
    auto padded = std::make_unique<PaddedCounter[]>(threads);
    countParallel(shared.get(), threads, 1000);
    countParallel(padded.get(), threads, 1000);
    bool ok = true;
    for (int t = 0; t < threads; ++t)
        ok = ok && shared[t].v.load() == 1000 && padded[t].v.load() == 1000;
    check(ok, "countParallel counts correctly");

    std::printf("    sizeof(Counter) = %zu, sizeof(PaddedCounter) = %zu\n", sizeof(Counter), sizeof(PaddedCounter));
    const double msShared = timeMs([&] { countParallel(shared.get(), threads, iters); }, 3);
    const double msPadded = timeMs([&] { countParallel(padded.get(), threads, iters); }, 3);
    std::printf("    %-34s %9.3f ms\n", "Counter (adjacent, shared lines)", msShared);
    std::printf("    %-34s %9.3f ms\n", "PaddedCounter (one per line)", msPadded);
}

void testTranspose()
{
    std::puts("\n== 8.7: cache blocking (transpose)");

    {
        const int n = 64;
        const auto in = randomFloats(n * n, 9);
        std::vector<float> a(n * n, -1.0f), b(n * n, -1.0f);
        transposeNaive(in.data(), a.data(), n);
        transposeBlocked<16>(in.data(), b.data(), n);
        bool okA = true, okB = true;
        for (int r = 0; r < n; ++r)
            for (int c = 0; c < n; ++c) {
                okA = okA && a[c * n + r] == in[r * n + c];
                okB = okB && b[c * n + r] == in[r * n + c];
            }
        check(okA, "transposeNaive");
        check(okB, "transposeBlocked");
    }

    const int n = 4096; // 64 MB per matrix
    const auto in = randomFloats(static_cast<std::size_t>(n) * n, 10);
    std::vector<float> out(static_cast<std::size_t>(n) * n);
    const double bytes = 2.0 * n * n * sizeof(float);
    row("transposeNaive", timeMs([&] { transposeNaive(in.data(), out.data(), n); }, 3), bytes);
    row("transposeBlocked<8>", timeMs([&] { transposeBlocked<8>(in.data(), out.data(), n); }, 3), bytes);
    row("transposeBlocked<32>", timeMs([&] { transposeBlocked<32>(in.data(), out.data(), n); }, 3), bytes);
    row("transposeBlocked<128>", timeMs([&] { transposeBlocked<128>(in.data(), out.data(), n); }, 3), bytes);
}

} // namespace

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    testSums();
    testAreas();
    testThreads();
    testFalseSharing();
    testTranspose();

    std::printf("\n%s\n", g_failures == 0 ? "All Milestone 8 tests pass." : "Some tests failed.");
    return g_failures == 0 ? 0 : 1;
}
