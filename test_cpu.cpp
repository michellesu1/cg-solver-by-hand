// Tests for Milestones 1, 2 and the CPU half of 6. Runs on your Mac.
//   make cpu
// Don't edit this file.

#include "cpu_analysis.h"
#include "csr.h"
#include "mesh.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const char* name)
{
    std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok)
        ++g_failures;
}

bool near(double a, double b, double tol = 1e-12) { return std::abs(a - b) <= tol; }

void testMilestone1()
{
    std::puts("== Milestone 1: Mesh<Real>");

    // These are constexpr variables, so triangleArea must work at compile time.
    constexpr double a = triangleArea(0.0, 0.0, 1.0, 0.0, 0.0, 1.0);
    check(near(a, 0.5), "constexpr triangleArea");
    constexpr double cw = triangleArea(0.0, 0.0, 0.0, 1.0, 1.0, 0.0);
    check(near(cw, 0.5), "triangleArea is positive for clockwise triangles");
    constexpr float af = triangleArea(0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 2.0f);
    check(af == 2.0f, "triangleArea<float>");
    check(absVal(-3) == 3 && absVal(2.5) == 2.5, "absVal");

    Mesh<double> m;
    m.reserve(3, 1);
    const int v0 = m.addVertex(0, 0);
    const int v1 = m.addVertex(1, 0);
    const int v2 = m.addVertex(0, 1);
    check(v0 == 0 && v1 == 1 && v2 == 2, "addVertex returns the new index");
    m.addTriangle(v0, v1, v2);
    check(m.numVertices() == 3 && m.numTris() == 1, "numVertices / numTris");
    check(m.x(1) == 1.0 && m.y(2) == 1.0, "x() / y()");
    check(m.tri(0) == std::array<int, 3>{ 0, 1, 2 }, "tri()");
    check(near(m.triArea(0), 0.5), "triArea");

    const Mesh<double> m2{ { 0, 2, 0 }, { 0, 0, 2 }, { 0, 1, 2 } };
    check(m2.numVertices() == 3 && m2.numTris() == 1 && near(m2.triArea(0), 2.0),
          "constructor taking arrays");
    check(m2.xs().size() == 3 && m2.tris().size() == 3, "constructor stores the arrays");

    const Mesh<double> g = makeGridMesh<double>(2);
    const bool gridBuilt = g.numVertices() == 9 && g.numTris() == 8;
    check(gridBuilt, "makeGridMesh counts");
    check(gridBuilt && g.tri(0) == std::array<int, 3>{ 0, 1, 4 } &&
              g.tri(1) == std::array<int, 3>{ 0, 4, 3 },
          "makeGridMesh triangle order");
    check(gridBuilt && near(g.x(4), 0.5) && near(g.y(4), 0.5) && near(g.x(8), 1.0) &&
              near(g.y(8), 1.0),
          "makeGridMesh vertex positions");
    double sum = 0.0;
    for (int t = 0; t < g.numTris(); ++t)
        sum += g.triArea(t);
    check(near(sum, 1.0), "makeGridMesh covers the unit square");
}

void testMilestone2()
{
    std::puts("\n== Milestone 2: CPU analysis");

    Mesh<double> m;
    m.addVertex(0, 0);
    m.addVertex(1, 0);
    m.addVertex(0, 1);
    m.addVertex(2, 0);
    m.addVertex(2, 2);
    m.addTriangle(0, 1, 2); // area 0.5
    m.addTriangle(1, 3, 4); // area 1
    m.addTriangle(0, 3, 4); // area 2

    check(near(totalArea(m), 3.5), "totalArea");
    check(largestTri(m) == 2, "largestTri");
    check(largestTri(Mesh<double>{}) == -1, "largestTri of an empty mesh is -1");
    check(trisByAreaDescending(m) == std::vector<int>{ 2, 1, 0 }, "trisByAreaDescending");
    check(countTrisBelow(m, 0.75) == 1 && countTrisBelow(m, 1.5) == 2, "countTrisBelow");

    const std::vector<double> areas = triAreasCpu(m);
    check(areas.size() == 3 && near(areas[0], 0.5) && near(areas[1], 1.0) && near(areas[2], 2.0),
          "triAreasCpu");

    const auto aos = toAoS(m);
    check(aos.size() == 3 && aos[1].x0 == 1 && aos[1].y0 == 0 && aos[1].x1 == 2 && aos[1].y1 == 0 &&
              aos[1].x2 == 2 && aos[1].y2 == 2,
          "toAoS");
    const auto soa = toSoA(m);
    check(soa.x0.size() == 3 && soa.y2.size() == 3 && soa.x2[2] == 2 && soa.y1[0] == 0 &&
              soa.y2[0] == 1,
          "toSoA");

    const Mesh<float> big = makeGridMesh<float>(1000);
    check(near(totalArea(big), 1.0, 1e-4), "totalArea<float> over 2M triangles (accumulator precision)");
}

void testMilestone6Cpu()
{
    std::puts("\n== Milestone 6 (CPU half): CSR Laplacian");

    const Mesh<double> g = makeGridMesh<double>(2);
    const CsrMatrix<double> L = buildLaplacian(g);
    check(L.rows == 9 && L.rowPtr.size() == 10, "rows and rowPtr size");
    check(L.nnz() == 41, "number of nonzeros (9 diagonal + 2 * 16 edges)");

    bool row0 = L.rowPtr.size() == 10 && L.rowPtr[0] == 0 && L.rowPtr[1] == 4 && L.nnz() >= 4 &&
                L.colIdx[0] == 0 && L.colIdx[1] == 1 && L.colIdx[2] == 3 && L.colIdx[3] == 4 &&
                L.values[0] == 3 && L.values[1] == -1 && L.values[2] == -1 && L.values[3] == -1;
    check(row0, "row 0 is [3 at col 0, -1 at cols 1, 3, 4]");

    bool sorted = L.rowPtr.size() == 10;
    for (std::size_t r = 0; sorted && r + 1 < L.rowPtr.size(); ++r)
        for (auto k = static_cast<std::size_t>(L.rowPtr[r]) + 1; k < static_cast<std::size_t>(L.rowPtr[r + 1]); ++k)
            if (L.colIdx[k - 1] >= L.colIdx[k])
                sorted = false;
    check(sorted, "columns sorted within each row, no duplicates");

    const CsrMatrix<double> L3 = buildLaplacian(makeGridMesh<double>(3));
    check(L3.nnz() == 82, "nonzeros on a 3x3 grid");

    // Every row of a Laplacian sums to zero, so L * (1, 1, ..., 1) = 0.
    const std::vector<double> y = spmvCpu(L, std::vector<double>(9, 1.0));
    bool zero = y.size() == 9;
    for (double v : y)
        zero = zero && near(v, 0.0);
    check(zero, "spmvCpu: L * ones == 0");

    std::vector<double> e0(9, 0.0);
    e0[0] = 1.0;
    const std::vector<double> col0 = spmvCpu(L, e0);
    check(col0.size() == 9 && col0[0] == 3 && col0[1] == -1 && col0[3] == -1 && col0[4] == -1 &&
              col0[2] == 0 && col0[8] == 0,
          "spmvCpu: L * e0 picks out column 0");
}

} // namespace

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0); // unbuffered, so a crash can't hide earlier results

    testMilestone1();
    testMilestone2();
    testMilestone6Cpu();

    std::printf("\n%s\n", g_failures == 0 ? "All CPU tests pass." : "Some tests failed.");
    return g_failures == 0 ? 0 : 1;
}
