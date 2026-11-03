// Milestone 1: Mesh<Real>
//
// Practices: classes, constructors + member initializer lists, std::move,
// const member functions, constexpr functions, templates, static_cast.
//
// Fill in every TODO 1.x, then run `make cpu`.

#pragma once

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

// Given. When this header is compiled by nvcc, functions marked HOST_DEVICE
// can be called from both CPU code and GPU kernels. Under clang++ on your Mac
// the macro expands to nothing.
#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

// TODO 1.1: constexpr absolute value.
// std::abs isn't constexpr before C++23 and can't be called from a GPU kernel,
// so write your own using the ?: operator.
template <typename Real>
HOST_DEVICE constexpr Real absVal(Real v)
{

    return v > Real{0} ? v : -v;

}

// TODO 1.2: constexpr area of a triangle from its three corners.
//   area = 0.5 * |(x1 - x0)(y2 - y0) - (x2 - x0)(y1 - y0)|
// Use absVal. Careful: the literal 0.5 is a double. What happens when Real is
// float? (Build with -Wconversion and the compiler will tell you.)
template <typename Real>
// tempalte header, followed by function
// type of T (typename) is deduced by argments. can be different across calls to this func.
// manual instantiation  float: triangleArea<float>(0, 0, 2, 0, ...) is implicit conversion, don't need static_cast
HOST_DEVICE constexpr Real triangleArea(Real x0, Real y0, Real x1, Real y1, Real x2, Real y2)
{
    //template argument
    const Real abs_input = (x1 - x0)*(y2 - y0) - (x2 - x0)*(y1 - y0);
    // !!add const bc doesnt change
    return Real{0.5} * absVal(abs_input);
    // !!need Real{0.5}, without this 0.5d makes the answer double and there's precision loss 
    // !!float 4 bytes and double 8 bytes
    // !!Real{0.5} is a temporary/rvalue so doesn't take extra mem. u can also static_cast<Real>(0.5)


}
template <typename Real>
class Mesh
{
private:
    std::vector<Real> m_x;    // vertex x coordinates (SoA layout)
    std::vector<Real> m_y;    // vertex y coordinates
    std::vector<int>  m_tris; // 3 vertex indices per triangle, flattened

public:
    Mesh() = default;

    // TODO 1.3: take ownership of existing arrays.
    // Use a member initializer list and std::move each parameter into its
    // member, so the arrays are moved rather than copied a second time.

    // !!putting x y tris as parameters already maeks a copy, without std::move 
    // !! there would be another copy. u could use pointers but still better to make copy usually
    Mesh(std::vector<Real> x, std::vector<Real> y, std::vector<int> tris)
    : m_x{std::move(x)}, m_y{std::move(y)}, m_tris{std::move(tris)}
    // colon for member initiliazation only
    {}

    // TODO 1.4: reserve capacity in all three vectors, so adding vertices and
    // triangles later never reallocates.
    void reserve(int numVerts, int numTris)
    {
        m_x.reserve(numVerts);
        m_y.reserve(numVerts);
        m_tris.reserve(3*numTris);
        //only doubles if it grows beyond reserve
    }

    // TODO 1.5: append a vertex and return its index.
    int addVertex(Real x, Real y)
    {
        m_x.push_back(x);
        m_y.push_back(y);
        return static_cast<int>(m_x.size()-1); //cast to int
    }

    // TODO 1.6: append a triangle.
    void addTriangle(int a, int b, int c)
    {
        m_tris.push_back(a);
        m_tris.push_back(b);
        m_tris.push_back(c);
    }

    // TODO 1.7: counts. .size() returns std::size_t, so use static_cast<int>.
    int numVertices() const { return static_cast<int>(m_x.size()); }
    int numTris() const { return static_cast<int>((m_tris.size()/3)); }

    // TODO 1.8: accessors.
    Real x(int v) const { return m_x[v]; }
    Real y(int v) const { return m_y[v]; } // already right type
    std::array<int, 3> tri(int t) const { return { m_tris[3*t], m_tris[3*t+1], m_tris[3*t+2] }; }
    // arrays on stack not heap, arrays faster and don't need to allocate space each time


    // TODO 1.9: area of triangle t. Use triangleArea.
    Real triArea(int t) const { 
        auto [a, b, c] = tri(t); //structured binding; auto a = tri(t)[0], tri calle donce
        // const auto& needed for binding reference to temporary, modifying temporary is mistake
        return triangleArea(x(a), y(a), x(b), y(b), x(c), y(c));

     }

    // Given: read-only access to the raw arrays, used for GPU upload later.
    // They return const references, so calling them never copies the arrays.
    const std::vector<Real>& xs() const { return m_x; }
    const std::vector<Real>& ys() const { return m_y; }
    const std::vector<int>& tris() const { return m_tris; }
};

// TODO 1.10: build an n x n grid over the unit square, 2 triangles per cell.
//
//   Vertices: (n+1)*(n+1) of them. Vertex (i, j) sits at (i/n, j/n) and has
//             index j*(n+1) + i. Add them with j in the outer loop.
//   Triangles: loop over cells with j outer, i inner. For cell (i, j) let
//             a = vertex(i, j),     b = vertex(i+1, j),
//             c = vertex(i+1, j+1), d = vertex(i, j+1)
//             and add triangle (a, b, c), then triangle (a, c, d).
//
//   Call reserve() first. Convert i, j and n with static_cast<Real> before
//   dividing (what would i / n give you with ints?).
template <typename Real>
Mesh<Real> makeGridMesh(int n)
//origin is bottom left. cannot deduce Real from args
{
    Mesh<Real> m;
    m.reserve((n+1)*(n+1), 2*n*n);

    //int num_vertices = (n+1)*(n+1);
    for (int j = 0; j <= n; j++) {
        Real cur_y = static_cast<Real>(j)/ static_cast<Real>(n);
        // cast not do Real{} becase big ints can be greater than Real
        //div slow do once outside
        //ROW MAJOR J OUTER I INNER
        for (int i = 0; i <= n; i++) {
            Real cur_x = static_cast<Real>(i)/ static_cast<Real>(n);
            m.addVertex(cur_x, cur_y);
            //priv so can't edit directly
                        
        }
    }

    auto get_index = [n](int i, int j) {return j*(n+1) + i;};
    //lambda auto type, indices not valeus so i and j better
    // &n can dangle

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            int a = get_index(i, j);
            int b = get_index(i+1, j);
            int c = get_index(i+1, j+1);
            int d = get_index(i, j+1);

            m.addTriangle(a, b, c);
            m.addTriangle(a, c, d);

        }
    }
    return m;
    // Copy elision = the compiler skips a copy or move entirely by building the object directly where it will end up.
    // or returning local will move auto

}
