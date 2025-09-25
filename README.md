# 🚀 CUDA-Accelerated Spatial Distance Histogram  

## 📌 Project Overview  
This project explores **GPU acceleration** for large-scale data analysis by computing the **Spatial Distance Histogram (SDH)** of 3D points. A naïve CPU implementation is compared against multiple GPU implementations to evaluate how **parallelization, memory coalescing, and shared memory** improve performance.  

Inspired by:  
*Pitaksirianan, Nouri, and Tu. "Efficient 2-Body Statistics Computation on GPUs: Parallelization & Beyond."*  

---

## ⚡ Key Contributions  
- Implemented **CPU baseline** and **GPU kernels** for pairwise distance computations.  
- Optimized memory layout (struct → split x/y/z arrays) to enable **global memory coalescing**.  
- Designed two GPU approaches:  
  1. **Shared Memory Kernel** – block-level histogram accumulation.  
  2. **Enhanced Block-Based Kernel** – reduced global memory traffic, register caching, intra/inter-block distance computation.  
- Used **constant memory** and **atomic operations** for efficient histogram updates.  

---

## 📊 Results  
Benchmarked with datasets of **10k, 50k, 100k, and 500k points**.  

- **GPU implementations significantly outperform CPU baselines.**  
- Transition from struct-based to split-array storage improved memory throughput.  
- **Optimized GPU kernel (Alg3)** achieved the **highest speedups**, especially at scale (500k points).  

| Dataset Size | CPU (P1) | GPU (P1) | GPU (P2 Basic) | GPU (P2 Optimized) |
|--------------|----------|----------|----------------|--------------------|
| 10,000       | Slow     | ⚡ Faster | ⚡⚡            | ⚡⚡⚡ Best |
| 500,000      | ❌ Impractical | ⚡ Moderate | ⚡⚡ Faster | ⚡⚡⚡ Best |  

*(Exact timings depend on hardware; trend is consistent.)*  

