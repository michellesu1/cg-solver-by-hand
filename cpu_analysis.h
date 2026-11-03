// Milestone 2: CPU analysis
//
// Practices: passing by const&, auto, range-based for, lambdas,
// std::sort / std::iota / std::count_if, returning containers by value,
// AoS vs SoA layouts.
//
// Fill in every TODO 2.x, then run `make cpu`.

#pragma once

#include "mesh.h"

#include <algorithm>
#include <numeric>
#include <vector>

// Given: the corner coordinates of every triangle, in two memory layouts.
// Milestone 4 uploads both to the GPU and compares how fast kernels read them.

// AoS (array of structs): one struct per triangle.
template <typename Real>
struct TriCoords
{
    Real x0, y0, x1, y1, x2, y2;
};

// SoA (struct of arrays): one array per coordinate.
template <typename Real>
struct TriCoordsSoA
{
    std::vector<Real> x0, y0, x1, y1, x2, y2;
};

// TODO 2.1: sum of all triangle areas.
// Accumulate in a double even when Real is float. The test sums 2 million
// small floats: try a float accumulator first and see what happens.
template <typename Real>
double totalArea(const Mesh<Real>& mesh)
{
    return 0.0;
}

// TODO 2.2: index of the largest triangle, or -1 if the mesh is empty.
template <typename Real>
int largestTri(const Mesh<Real>& mesh)
{
    return -1;
}

// TODO 2.3: all triangle indices, sorted by area, largest first.
// Fill a vector with 0, 1, 2, ... using std::iota, then std::sort it with a
// lambda comparator that captures the mesh by reference.
template <typename Real>
std::vector<int> trisByAreaDescending(const Mesh<Real>& mesh)
{
    return {};
}

// TODO 2.4: how many triangles have area strictly below `threshold`.
// Use std::count_if with a lambda. What does the lambda need to capture?
template <typename Real>
int countTrisBelow(const Mesh<Real>& mesh, Real threshold)
{
    return 0;
}

// TODO 2.5: the area of every triangle (the CPU reference the GPU results are
// checked against). Reserve before you push_back.
template <typename Real>
std::vector<Real> triAreasCpu(const Mesh<Real>& mesh)
{
    return {};
}

// TODO 2.6: corner coordinates of every triangle in AoS layout.
template <typename Real>
std::vector<TriCoords<Real>> toAoS(const Mesh<Real>& mesh)
{
    return {};
}

// TODO 2.7: corner coordinates of every triangle in SoA layout.
template <typename Real>
TriCoordsSoA<Real> toSoA(const Mesh<Real>& mesh)
{
    return {};
}
