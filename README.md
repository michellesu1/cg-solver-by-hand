# GPU mesh toolkit

Build a triangle mesh in C++, move it to the GPU, and run analysis kernels on it, timing everything.

You write every function body. The headers have the signatures plus `TODO` comments, and the tests tell you what passes. Don't edit the test files.

## Files

| File | Milestone | You write |
|---|---|---|
| `mesh.h` | 1 | `Mesh<Real>` class, `constexpr` area helper, grid generator |
| `cpu_analysis.h` | 2 | analysis functions, AoS/SoA conversion |
| `device_buffer.cuh` | 3 | `DeviceBuffer<T>` and `GpuTimer` (RAII classes) |
| `kernels.cuh` | 4, 5, 6 | area kernels, three reductions, CSR SpMV |
| `csr.h` | 6 | building a sparse matrix, CPU SpMV |
| `RESULTS.md` | 7 | your timings and explanations |
| `cpu_perf.h` | 8 (optional) | CPU performance: accumulators, SIMD, threads, false sharing, cache blocking |
| `test_cpu.cpp`, `test_gpu.cu`, `bench_cpu.cpp` | | tests (given) |

## Milestones

| # | What | Practices |
|---|---|---|
| 1 | `Mesh<Real>` | classes, constructors, initializer lists, `std::move`, `const` member functions, `constexpr`, templates, `static_cast` |
| 2 | CPU analysis | `const&`, `auto`, range-for, lambdas, `<algorithm>`, AoS vs SoA, float precision |
| 3 | `DeviceBuffer<T>`, `GpuTimer` | RAII, destructors, `= delete`, move constructor/assignment, `noexcept`, `explicit` |
| 4 | Area kernels | kernel launch, global index, grid-stride loops, coalescing (SoA vs AoS) |
| 5 | Reductions | atomics, shared memory, `__syncthreads`, warp shuffles, non-type template parameters |
| 6 | CSR SpMV | sparse formats, irregular memory access, memory-bound kernels (CPU half in `csr.h`) |
| 7 | Write-up | bandwidth vs peak, explaining performance |
| 8 | CPU performance (optional) | dependency chains, auto-vectorization, `__restrict__`, `std::thread`, false sharing, cache blocking |

Suggested order: 1 → 2 → 6 (CPU half, `csr.h`) on your Mac, then 3 → 4 → 5 → 6 (GPU half) on Colab, then 7.

## Running

**On your Mac (Milestones 1, 2, 6 CPU half):**

```bash
make cpu
```

Warnings about unused parameters are expected until you fill in the TODOs. `-Wconversion` warnings are worth reading: they point at missing `static_cast`s.

**Milestone 8 (CPU performance), also on your Mac:**

```bash
make perf   # tests + timings, built with -O3
make vec    # which loops the compiler vectorized, and why not
```

**On Google Colab (Milestones 3-7).** Macs don't have NVIDIA GPUs, so the CUDA parts run on Colab's free GPU:

1. In Colab: Runtime → Change runtime type → T4 GPU.
2. Zip this folder and upload it with the file panel on the left.
3. Run these in notebook cells:
   ```
   !unzip -o mesh_project.zip
   %cd mesh_project
   !nvidia-smi
   !make gpu
   ```
4. Edit the `.cuh` files in Colab's editor (double-click a file in the file panel), then re-run `!make gpu`. Download your finished files at the end so you keep them.

On a GPU other than a T4, pass its architecture: `!make gpu ARCH=sm_80` (A100) or `ARCH=sm_89` (L4).

`./test_gpu 200` runs on a smaller mesh, which is handy while debugging.

## Getting unstuck

- Read the compiler error from the top: the first error is usually the real one.
- Add `printf` inside a kernel to debug it (works on the GPU), on a small mesh.
- If a kernel produces zeros, check that the launcher actually launches it.
- Reference: NVIDIA's *CUDA C++ Programming Guide* (chapters 1-5) and Mark Harris's *Optimizing Parallel Reduction in CUDA* slides.
