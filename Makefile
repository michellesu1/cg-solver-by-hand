CXXFLAGS = -std=c++20 -Wall -Wextra -Wconversion -Wno-sign-conversion -g
ARCH    ?= sm_75
NVFLAGS  = -std=c++17 -O3 -arch=$(ARCH) -Xcompiler -Wall

# Milestones 1, 2, 6 (CPU): runs on your Mac
cpu: test_cpu
	./test_cpu

test_cpu: test_cpu.cpp mesh.h cpu_analysis.h csr.h
	$(CXX) $(CXXFLAGS) test_cpu.cpp -o test_cpu

# Milestones 3-7: needs nvcc and an NVIDIA GPU (Colab T4 is sm_75; A100 is sm_80)
gpu: test_gpu
	./test_gpu

test_gpu: test_gpu.cu mesh.h cpu_analysis.h csr.h device_buffer.cuh kernels.cuh
	nvcc $(NVFLAGS) test_gpu.cu -o test_gpu

clean:
	rm -f test_cpu test_gpu

.PHONY: cpu gpu clean
