// Milestone 6 (CPU half): sparse matrices in CSR format
//
// Practices: building a data structure with std::vector, sort/unique,
// lambdas, returning by value.
//
// Fill in TODO 6.1 and 6.2, then run `make cpu`. The GPU half is in
// kernels.cuh.

#pragma once

#include "mesh.h"

#include <algorithm>
#include <utility>
#include <vector>

// Given. Compressed Sparse Row storage of a rows x rows matrix.
// The nonzeros of row r are at positions rowPtr[r] .. rowPtr[r+1]-1 of
// colIdx and values.
template <typename Real>
struct CsrMatrix
{
    int rows = 0;
    std::vector<int>  rowPtr; // size rows + 1, rowPtr[0] == 0
    std::vector<int>  colIdx; // size nnz, sorted ascending within each row
    std::vector<Real> values; // size nnz

    int nnz() const { return static_cast<int>(colIdx.size()); }
};

// TODO 6.1: the graph Laplacian of the mesh's vertices.
//
//   One row and one column per vertex. Two vertices are neighbors if some
//   triangle edge connects them.
//     L[i][i] = number of neighbors of i
//     L[i][j] = -1 if i and j are neighbors
//     everything else is 0 (not stored)
//
//   One approach: collect every edge in both directions as (row, col) pairs,
//   std::sort them, remove duplicates with std::unique + erase, then build
//   rowPtr / colIdx / values in one pass (the diagonal must be stored in
//   sorted column order too).
template <typename Real>
CsrMatrix<Real> buildLaplacian(const Mesh<Real>& mesh)
{
    return {};
}

// TODO 6.2: y = A * x on the CPU (the reference for the GPU kernel).
template <typename Real>
std::vector<Real> spmvCpu(const CsrMatrix<Real>& A, const std::vector<Real>& x)
{
    return {};
}
