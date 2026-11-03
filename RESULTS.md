# Results

GPU:
Mesh size:

## Timings

Paste the Milestone 7 table from `./test_gpu` here.

```
```

## Questions

Answer each in two or three sentences, in your own words. These are your interview talking points.

1. **SoA vs AoS.** Which area kernel was faster, and why? Talk about which addresses neighboring threads in a warp read.

2. **CPU vs GPU.** How much faster was the GPU area kernel than the CPU? Was the GPU limited by computation or by memory bandwidth? (Compare its GB/s with the peak.)

3. **Atomic reduction.** Why is the atomic version slow? Why is its sum also *less accurate* than the others? (Think about what happens when you add a tiny float to a running total near 1.0, two million times in a row.)

4. **Shared vs shuffle.** What does the warp-shuffle version avoid that the shared-memory version needs?

5. **`__syncthreads()`.** What goes wrong if one thread in a block skips a `__syncthreads()` that the others reach?

6. **SpMV.** Why does SpMV reach a lower fraction of peak bandwidth than the area kernel? Which memory accesses aren't coalesced?

7. **Next steps.** If you had another day, what would you optimize first, and how would you find out where time goes? (Name the profiler.)

8. **C++ design.** Why is `DeviceBuffer` move-only? What would break if it were copyable?
