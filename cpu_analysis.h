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
    //ONE TRIANGLE, need an array of these structs
};

// SoA (struct of arrays): one array per coordinate.
template <typename Real>
struct TriCoordsSoA
{
    std::vector<Real> x0, y0, x1, y1, x2, y2;

    //each one is a vector with that coordinate for all triangles
};

// TODO 2.1: sum of all triangle areas.
// Accumulate in a double even when Real is float. The test sums 2 million
// small floats: try a float accumulator first and see what happens.
template <typename Real>
double totalArea(const Mesh<Real>& mesh)
{
   int num_tris = mesh.numTris();
   double sum = 0.0;
   for (int i = 0; i < num_tris; i++) {
        sum += mesh.triArea(i);
   }

   return sum;

}

// TODO 2.2: index of the largest triangle, or -1 if the mesh is empty.
template <typename Real>
int largestTri(const Mesh<Real>& mesh)
{
    const int num_tris = mesh.numTris();
    if (num_tris == 0) {
        return -1;
    }

    Real maxa = Real{0}; //build real
    int max_i = 0;
    for (int i = 0; i < num_tris; i++) {
        Real cur = mesh.triArea(i);

        if (cur > maxa) {
            maxa = cur;
            max_i = i;
        }
    }
    return max_i;

}

// TODO 2.3: all triangle indices, sorted by area, largest first.
// Fill a vector with 0, 1, 2, ... using std::iota, then std::sort it with a
// lambda comparator that captures the mesh by reference.
template <typename Real>
std::vector<int> trisByAreaDescending_versionA(const Mesh<Real>& mesh)
{

    // heap: v's n ints
    // stack: v (pointer, size, capacity), lambda
    // registers: a, b, 2 Reals. gone every comparison

    std::vector<int> v(mesh.numTris());
    std::iota(v.begin(), v.end(), 0);
    std::sort(v.begin(), v.end(), [&mesh](int i, int j){
        //dont copy big
        return mesh.triArea(i) > mesh.triArea(j);
    });
    return v;
}

// version B, precompute sizes with 2.5
template <typename Real>
std::vector<int> trisByAreaDescending(const Mesh<Real>& mesh)
{

    // heap: v's n ints, areas n Reals
    // stack: areas, v (pointer, size, capacity), lambda
    // registers: a, b, gone every comparison

    std::vector<int> v(mesh.numTris());
    std::vector<Real> areas = triAreasCpu(mesh);
    // always call with ob directly, & and * in param in func def

    std::iota(v.begin(), v.end(), 0);
    std::sort(v.begin(), v.end(), [&areas](int i, int j){
        //dont copy big
        return areas[i] > areas[j];
    });
    return v;
}


// TODO 2.4: how many triangles have area strictly below `threshold`.
// Use std::count_if with a lambda. What does the lambda need to capture?

//this is just for practice easier/faster to do loop
template <typename Real>
int countTrisBelow(const Mesh<Real>& mesh, Real threshold)
{

    std::vector<int> v(mesh.numTris());
    std::iota(v.begin(), v.end(), 0);
    return static_cast<int>(
        std::count_if(v.begin(), v.end(), [&mesh, threshold](int i) {
            return mesh.triArea(i) < threshold;
        })
    );
}

// TODO 2.5: the area of every triangle (the CPU reference the GPU results are
// checked against). Reserve before you push_back.
template <typename Real>
std::vector<Real> triAreasCpu(const Mesh<Real>& mesh)
{
    //could also make v beforehand instead of pushback

    std::vector<Real> v; //default construct. int/built in types are old memory so garb
    v.reserve(mesh.numTris());
    for (int i = 0; i < mesh.numTris(); i++) {
        v.push_back(mesh.triArea(i));
    }
    return v;
}


// TODO 2.6: corner coordinates of every triangle in AoS layout.
template <typename Real>
std::vector<TriCoords<Real>> toAoS(const Mesh<Real>& mesh)
{
    // <xy xy xy>
    // reserve once and add new struct for each triangle
    std::vector<TriCoords<Real>> AoS; //could be AoS(n) to not do push_back, default to 0
    const int n{mesh.numTris()};
    AoS.reserve(n);
    for (int i = 0; i < n; i++) {
        auto [a, b, c] = mesh.tri(i); // std::array<int, 3> t{ mesh.tri(i) };

        AoS.emplace_back( // essent. builds directly in that slots memory instead of building new.
            // {} with push_back signifies building new
            mesh.x(a), mesh.y(a), mesh.x(b), mesh.y(b), mesh.x(c), mesh.y(c) 
        );

        /*

         AoS.push_back({
            mesh.x(a), mesh.y(a), mesh.x(b), mesh.y(b), mesh.x(c), mesh.y(c), 
        });
        
        AoS.push_back(TriCoords<Real>{
            .x0 = mesh.x(a), mesh.y(a), mesh.x(b), mesh.y(b), mesh.x(c), mesh.y(c), 
        });
        //MUST DESGINATE ALL NOT JUST ONE!!!!

        // the type is inferred because the destination AoS is a TriCoords<Real>
        // Aggregate = all-public data + no user constructor (struct or class) → fill with { } or { .x = ... }; add a constructor and you lose aggregate init/labels.
        
        */
    }
    return AoS;
}

// TODO 2.7: corner coordinates of every triangle in SoA layout.
template <typename Real>
TriCoordsSoA<Real> toSoA(const Mesh<Real>& mesh)
{
    // (x x x) ,,, y y y y 
    // reserve all and push for each triangle
    // soa more lines but faster for gpu to read

    TriCoordsSoA<Real> SoA;
    const int n{mesh.numTris()};
    SoA.x0.reserve(n); SoA.y0.reserve(n);
    SoA.x1.reserve(n); SoA.y1.reserve(n);
    SoA.x2.reserve(n); SoA.y2.reserve(n);

    for (int i = 0; i < n; i++) {
        auto [a, b, c] = mesh.tri(i);
        SoA.x0.emplace_back(mesh.x(a)); 
        SoA.y0.push_back(Real{mesh.y(a)}); //don't need Real because SoA.y0 is std::vector<Real>
        SoA.x1.emplace_back(mesh.x(b));
        SoA.y1.emplace_back(mesh.y(b));
        SoA.x2.emplace_back(mesh.x(c));
        SoA.y2.emplace_back(mesh.y(c));
    }

    return SoA;

    /* this is more to read

    When a temporary is used to initialize something of the same type, C++17 guarantees it gets built directly in the final spot. So each member is constructed directly: zero copies, zero moves.

    TriCoordsSoA<Real> soa{
        std::vector<Real>(n), std::vector<Real>(n),   // x0, y0
        std::vector<Real>(n), std::vector<Real>(n),   // x1, y1
        std::vector<Real>(n), std::vector<Real>(n)    // x2, y2
    };
    
    */


}
