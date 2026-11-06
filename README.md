# GPU CG thermal solver

Build a simple triangle mesh and CG solver. Write every function body by hand. The headers have the signatures plus `TODO` comments, and the tests tell you what passes. Don't edit the test files.

Discretizing heat conduction on a triangle mesh gives a sparse graph Laplacian (CSR), so explicit time steps reduce to repeated SpMV and steady-state solves reduce to Conjugate Gradient: SpMV plus dot-product reductions and axpy updates. The C++ side builds the CSR matrix. The CUDA side covers the kernels a solver depends on: coalesced SoA, reductions, shared memory, and a row-parallel SpMV whose indirect gathers expose the cost of irregular meshes. Every kernel is validated against a CPU reference within float tolerance and benchmarked against the L40S's ~864 GB/s peak. Together they form the core of a GPU CG thermal solver.

## Files

| File | Milestone | You write |
|---|---|---|
| `mesh.h` | 1 | `Mesh<Real>` class, `constexpr` area helper, grid generator |
| `cpu_analysis.h` | 2 | analysis functions, AoS/SoA conversion |
| `device_buffer.cuh` | 3 | `DeviceBuffer<T>` and `GpuTimer` (RAII classes) |
| `kernels.cuh` | 4, 5, 6 | area kernels, three reductions, CSR SpMV |
| `csr.h` | 6 | building a sparse matrix, CPU SpMV |
| `solver.cuh` (TODO) | 7-10 | heat stepping, axpy/dot kernels, Conjugate Gradient |
| `cpu_perf.h` (TODO) | 11 (optional) | CPU performance: accumulators, SIMD, threads, false sharing, cache blocking |
| `RESULTS.md` | 12 | your timings and explanations |
| `test_cpu.cpp`, `test_gpu.cu`, `bench_cpu.cpp` | | tests (given) |

## Milestones

### Triangle mesh and kernels

| # | What | Practices |
|---|---|---|
| 1 | `Mesh<Real>` | classes, constructors, initializer lists, `std::move`, `const` member functions, `constexpr`, templates, `static_cast` |
| 2 | CPU analysis | `const&`, `auto`, range-for, lambdas, `<algorithm>`, AoS vs SoA, float precision |
| 3 | `DeviceBuffer<T>`, `GpuTimer` | RAII, destructors, `= delete`, move constructor/assignment, `noexcept`, `explicit` |
| 4 | Area kernels | kernel launch, global index, grid-stride loops, coalescing (SoA vs AoS) |
| 5 | Reductions | atomics, shared memory, `__syncthreads`, warp shuffles, non-type template parameters |
| 6 | (TODO) CSR SpMV | sparse formats, irregular memory access, memory-bound kernels (CPU half in `csr.h`) |

### Solver

| # | What | Practices |
|---|---|---|
| 7 | (TODO) Explicit heat diffusion: `T ← T − dt·α·(L·T) + dt·Q` | SpMV + axpy kernel per step, stability limit on `dt` (show it blowing up when `dt` is too large), total heat as a reduction |
| 8 | (TODO) Conjugate Gradient on `(L + εI)·x = b` | `axpy` and `dot` kernels (reuse the M5 reduction), the CG loop on the GPU, residual-based stopping, iteration counts vs mesh size (~√κ) |
| 9 | (TODO) Steady-state temperature with fixed-temperature boundaries | Dirichlet boundary conditions (fixed vertices move into `b`, interior stays SPD), validate against a 1D/analytic case, mesh refinement check |
| 10 | (TODO) Jacobi-preconditioned CG (optional) | preconditioning, iterations vs condition number, mixed precision (float SpMV, double dot products) |

### Performance and write-up

| # | What | Practices |
|---|---|---|
| 11 | CPU performance (optional) | dependency chains, auto-vectorization, `__restrict__`, `std::thread`, false sharing, cache blocking |
| 12 | Write-up (`RESULTS.md`) | bandwidth vs peak, explaining performance, solver convergence |

Suggested order: 1 → 2 → 6 (CPU half, `csr.h`) on your Mac, then 3 → 4 → 5 → 6 (GPU half) on MIT Engaging, then 7 → 8 → 9 → 10, then 12.

## Running

**On your Mac (Milestones 1, 2, 6 CPU half):**

```bash
make cpu
```

Warnings about unused parameters are expected until you fill in the TODOs. `-Wconversion` warnings are worth reading: they point at missing `static_cast`s.

**Milestone 11 (CPU performance), also on your Mac:**

```bash
make perf   # tests + timings, built with -O3
make vec    # which loops the compiler vectorized, and why not
```

**On MIT Engaging (Milestones 3-10).** Macs don't have NVIDIA GPUs, so the CUDA parts run on an Engaging GPU node:

1. Clone the repo on Engaging (once): `git clone git@github.com:michellesu1/triangle-mesh-gpu.git ~/triangle-mesh-gpu`
2. Get a GPU node and load CUDA:
   ```
   salloc -p mit_normal_gpu --gres=gpu:1 -t 01:00:00
   module load cuda/12.9.1
   cd ~/triangle-mesh-gpu
   nvidia-smi
   make gpu
   ```
   (`sinfo -o "%P %G" | grep -i gpu` lists GPU partitions if that one is rejected; `module spider cuda` lists CUDA versions. The 12.4 modules need `deprecated-modules`.)
3. Edit the `.cuh` files, re-run `make gpu`, then `git commit` and `git push` so your Mac can `git pull` the work.

The default `ARCH` is `sm_89` (L40S). On another GPU, pass its architecture: `make gpu ARCH=sm_80` (A100), `ARCH=sm_90` (H100), or `ARCH=sm_70` (V100).

`./test_gpu 200` runs on a smaller mesh, which is handy while debugging.

## Getting unstuck

- Read the compiler error from the top: the first error is usually the real one.
- Add `printf` inside a kernel to debug it (works on the GPU), on a small mesh.
- If a kernel produces zeros, check that the launcher actually launches it.
- Reference: NVIDIA's *CUDA C++ Programming Guide* (chapters 1-5), Mark Harris's *Optimizing Parallel Reduction in CUDA* slides, and Shewchuk's *An Introduction to the Conjugate Gradient Method Without the Agonizing Pain* (for 8-10).
