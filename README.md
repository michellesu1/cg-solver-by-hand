# GPU mesh toolkit exercise

Build a triangle mesh in C++ and move it to the GPU.

You write every function body. The headers have the signatures plus `TODO` comments, and the tests tell you what passes. Don't edit the test files.

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

Suggested order: 1 → 2 → 6 (CPU half, `csr.h`) on your Mac, then 3 → 4 → 5 → 6 (GPU half) on MIT Engaging, then 7.

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

**On MIT Engaging (Milestones 3-7).** Macs don't have NVIDIA GPUs, so the CUDA parts run on an Engaging GPU node:

1. Clone the repo on Engaging (once): `git clone git@github.com:michellesu1/triangle-mesh-gpu.git ~/triangle-mesh-gpu`
2. Get a GPU node and load CUDA:
   ```
   salloc -p mit_normal_gpu --gres=gpu:1 -t 01:00:00
   module load cuda
   cd ~/triangle-mesh-gpu
   nvidia-smi
   make gpu
   ```
   (`sinfo -o "%P %G" | grep -i gpu` lists GPU partitions if that one is rejected; `module avail cuda` lists CUDA versions.)
3. Edit the `.cuh` files, re-run `make gpu`, then `git commit` and `git push` so your Mac can `git pull` the work.

The default `ARCH` is `sm_89` (L40S). On another GPU, pass its architecture: `make gpu ARCH=sm_80` (A100), `ARCH=sm_90` (H100), or `ARCH=sm_70` (V100).

`./test_gpu 200` runs on a smaller mesh, which is handy while debugging.

## Getting unstuck

- Read the compiler error from the top: the first error is usually the real one.
- Add `printf` inside a kernel to debug it (works on the GPU), on a small mesh.
- If a kernel produces zeros, check that the launcher actually launches it.
- Reference: NVIDIA's *CUDA C++ Programming Guide* (chapters 1-5) and Mark Harris's *Optimizing Parallel Reduction in CUDA* slides.
