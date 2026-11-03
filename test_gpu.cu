// Tests and benchmarks for Milestones 3-7. Needs an NVIDIA GPU (see README).
//   make gpu            then   ./test_gpu [n]
// n is the grid size (default 1000, which gives 2 million triangles).
// Don't edit this file.

#include "cpu_analysis.h"
#include "csr.h"
#include "device_buffer.cuh"
#include "kernels.cuh"
#include "mesh.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const char* name)
{
    std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok)
        ++g_failures;
}

struct Result
{
    std::string name;
    double ms;
    double bytes; // bytes the kernel must read + write at minimum
};
std::vector<Result> g_results;

constexpr int REPS = 20;

void syncAndCheck()
{
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Runs before() + launch() once to warm up, then REPS more times, timing only
// launch(). Returns the average time in milliseconds.
template <typename Before, typename Launch>
double timeGpu(Before before, Launch launch)
{
    GpuTimer timer;
    before();
    launch();
    syncAndCheck();

    double total = 0.0;
    for (int r = 0; r < REPS; ++r) {
        before();
        timer.start();
        launch();
        timer.stop();
        total += timer.elapsedMs();
    }
    syncAndCheck();
    return total / REPS;
}

template <typename F>
double timeCpu(F work)
{
    const auto t0 = std::chrono::steady_clock::now();
    work();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

bool allClose(const std::vector<float>& got, const std::vector<float>& want, double rtol, double atol)
{
    if (got.size() != want.size())
        return false;
    for (std::size_t i = 0; i < got.size(); ++i)
        if (std::abs(got[i] - want[i]) > rtol * std::abs(want[i]) + atol)
            return false;
    return true;
}

bool stopIfFailed(const char* milestone)
{
    if (g_failures == 0)
        return false;
    std::printf("\nFix %s before moving on.\n", milestone);
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    const int n = (argc > 1) ? std::atoi(argv[1]) : 1000;

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    int memClockKHz = 0;
    int busWidthBits = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&busWidthBits, cudaDevAttrGlobalMemoryBusWidth, device));
    const double peakGBs = 2.0 * memClockKHz * 1e3 * (busWidthBits / 8.0) / 1e9;
    std::printf("GPU: %s, theoretical peak bandwidth ~%.0f GB/s\n", prop.name, peakGBs);
    std::printf("Mesh: %d x %d grid\n\n", n, n);

    const Mesh<float> mesh = makeGridMesh<float>(n);
    if (mesh.numTris() != 2 * n * n) {
        std::puts("makeGridMesh isn't finished. Get `make cpu` passing first.");
        return 1;
    }
    const int nt = mesh.numTris();
    const auto ntu = static_cast<std::size_t>(nt);

    // ------------------------------------------------------------------------
    std::puts("== Milestone 3: DeviceBuffer and GpuTimer");
    check(!std::is_copy_constructible_v<DeviceBuffer<float>>, "DeviceBuffer can't be copy-constructed");
    check(!std::is_copy_assignable_v<DeviceBuffer<float>>, "DeviceBuffer can't be copy-assigned");
    check(std::is_nothrow_move_constructible_v<DeviceBuffer<float>>, "noexcept move constructor");
    check(std::is_nothrow_move_assignable_v<DeviceBuffer<float>>, "noexcept move assignment");
    check(!std::is_convertible_v<std::size_t, DeviceBuffer<float>>, "size constructor is explicit");
    {
        std::vector<float> host(1000);
        for (std::size_t i = 0; i < host.size(); ++i)
            host[i] = static_cast<float>(i);

        DeviceBuffer<float> a(host.size());
        check(a.size() == host.size() && a.data() != nullptr, "constructor allocates");
        a.upload(host);

        DeviceBuffer<float> b{ std::move(a) };
        check(a.data() == nullptr && a.size() == 0 && b.size() == host.size(),
              "move constructor transfers ownership");

        DeviceBuffer<float> c;
        c = std::move(b);
        check(b.data() == nullptr && b.size() == 0 && c.size() == host.size(),
              "move assignment transfers ownership");

        std::vector<float> back;
        c.download(back);
        check(back == host, "upload/download round trip survives two moves");
    }
    {
        DeviceBuffer<float> big(ntu);
        GpuTimer timer;
        timer.start();
        CUDA_CHECK(cudaMemset(big.data(), 0, ntu * sizeof(float)));
        timer.stop();
        check(timer.elapsedMs() > 0.0f, "GpuTimer measures time");
    }
    if (stopIfFailed("Milestone 3"))
        return 1;

    // ------------------------------------------------------------------------
    std::puts("\n== Milestone 4: area kernels, SoA vs AoS");
    std::vector<float> refAreas;
    const double cpuAreaMs = timeCpu([&] { refAreas = triAreasCpu(mesh); });
    const TriCoordsSoA<float> soa = toSoA(mesh);
    const std::vector<TriCoords<float>> aos = toAoS(mesh);
    if (refAreas.size() != ntu || soa.x0.size() != ntu || aos.size() != ntu) {
        std::puts("Milestone 2 isn't finished. Get `make cpu` passing first.");
        return 1;
    }

    DeviceBuffer<float> dx0(ntu), dy0(ntu), dx1(ntu), dy1(ntu), dx2(ntu), dy2(ntu);
    dx0.upload(soa.x0);
    dy0.upload(soa.y0);
    dx1.upload(soa.x1);
    dy1.upload(soa.y1);
    dx2.upload(soa.x2);
    dy2.upload(soa.y2);
    DeviceBuffer<TriCoords<float>> dAos(ntu);
    dAos.upload(aos);
    DeviceBuffer<float> dAreas(ntu);
    std::vector<float> gpuAreas;
    const double areaBytes = static_cast<double>(nt) * 7 * sizeof(float);
    auto clearAreas = [&] { CUDA_CHECK(cudaMemset(dAreas.data(), 0, ntu * sizeof(float))); };

    double ms = timeGpu(clearAreas, [&] {
        launchAreaSoA(dx0.data(), dy0.data(), dx1.data(), dy1.data(), dx2.data(), dy2.data(),
                      dAreas.data(), nt);
    });
    dAreas.download(gpuAreas);
    check(allClose(gpuAreas, refAreas, 1e-4, 1e-12), "SoA kernel matches the CPU");
    g_results.push_back({ "area SoA", ms, areaBytes });

    ms = timeGpu(clearAreas, [&] { launchAreaAoS(dAos.data(), dAreas.data(), nt); });
    dAreas.download(gpuAreas);
    check(allClose(gpuAreas, refAreas, 1e-4, 1e-12), "AoS kernel matches the CPU");
    g_results.push_back({ "area AoS", ms, areaBytes });
    g_results.push_back({ "area CPU (1 core)", cpuAreaMs, areaBytes });
    if (stopIfFailed("Milestone 4"))
        return 1;

    // ------------------------------------------------------------------------
    std::puts("\n== Milestone 5: reductions (summing the areas from Milestone 4)");
    const double exactSum = totalArea(mesh);
    DeviceBuffer<float> dSum(1);
    const double sumBytes = static_cast<double>(nt) * sizeof(float);
    auto clearSum = [&] { CUDA_CHECK(cudaMemset(dSum.data(), 0, sizeof(float))); };

    auto runReduction = [&](const char* name, void (*launch)(const float*, float*, int), double tol) {
        const double t = timeGpu(clearSum, [&] { launch(dAreas.data(), dSum.data(), nt); });
        std::vector<float> h;
        dSum.download(h);
        const double got = h.empty() ? 0.0 : h[0];
        const double relErr = std::abs(got - exactSum) / exactSum;
        check(relErr <= tol, name);
        std::printf("      sum = %.7f, exact = %.7f, relative error = %.1e\n", got, exactSum, relErr);
        g_results.push_back({ name, t, sumBytes });
    };
    // The atomic version gets a loose tolerance on purpose: look at its error
    // and explain it in RESULTS.md.
    runReduction("reduce atomic", launchReduceAtomic, 0.1);
    runReduction("reduce shared", launchReduceShared, 1e-3);
    runReduction("reduce shuffle", launchReduceShuffle, 1e-3);
    if (stopIfFailed("Milestone 5"))
        return 1;

    // ------------------------------------------------------------------------
    std::puts("\n== Milestone 6: CSR SpMV (graph Laplacian of the mesh)");
    const CsrMatrix<float> L = buildLaplacian(mesh);
    if (L.rows != mesh.numVertices()) {
        std::puts("buildLaplacian isn't finished. Get `make cpu` passing first.");
        return 1;
    }
    const auto rowsU = static_cast<std::size_t>(L.rows);
    std::vector<float> x(rowsU);
    for (std::size_t i = 0; i < rowsU; ++i)
        x[i] = static_cast<float>((i * 2654435761u) % 1000u) / 1000.0f;

    std::vector<float> yRef;
    const double cpuSpmvMs = timeCpu([&] { yRef = spmvCpu(L, x); });

    DeviceBuffer<int> dRowPtr(L.rowPtr.size());
    DeviceBuffer<int> dColIdx(L.colIdx.size());
    DeviceBuffer<float> dValues(L.values.size());
    DeviceBuffer<float> dX(rowsU);
    DeviceBuffer<float> dY(rowsU);
    dRowPtr.upload(L.rowPtr);
    dColIdx.upload(L.colIdx);
    dValues.upload(L.values);
    dX.upload(x);

    ms = timeGpu([&] { CUDA_CHECK(cudaMemset(dY.data(), 0, rowsU * sizeof(float))); },
                 [&] {
                     launchSpmv(L.rows, dRowPtr.data(), dColIdx.data(), dValues.data(), dX.data(),
                                dY.data());
                 });
    std::vector<float> yGpu;
    dY.download(yGpu);
    check(allClose(yGpu, yRef, 1e-4, 1e-5), "SpMV kernel matches the CPU");
    const double spmvBytes = (static_cast<double>(L.rows) + 1) * sizeof(int) +
                             static_cast<double>(L.nnz()) * (sizeof(int) + 2 * sizeof(float)) +
                             static_cast<double>(L.rows) * sizeof(float);
    g_results.push_back({ "spmv GPU", ms, spmvBytes });
    g_results.push_back({ "spmv CPU (1 core)", cpuSpmvMs, spmvBytes });
    if (stopIfFailed("Milestone 6"))
        return 1;

    // ------------------------------------------------------------------------
    std::puts("\n== Milestone 7: results (copy this table into RESULTS.md)");
    std::printf("%-20s %12s %10s %8s\n", "kernel", "time (ms)", "GB/s", "% peak");
    for (const auto& r : g_results) {
        const double gbs = r.ms > 0.0 ? r.bytes / (r.ms * 1e6) : 0.0;
        std::printf("%-20s %12.4f %10.1f %7.0f%%\n", r.name.c_str(), r.ms, gbs, 100.0 * gbs / peakGBs);
    }
    std::puts("\nAll GPU tests pass.");
    return 0;
}
