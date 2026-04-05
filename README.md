# GPU-Accelerated Histogram Equalisation
- Course: CMP3752 – Parallel Programming
- Institution: University of Lincoln
- Grade Received: 69%
- Language: C++ / OpenCL
- Target Hardware: GPU (parallel processing)

# Overview
This project implements digital image enhancement using the histogram equalisation algorithm on parallel hardware. The program is written in OpenCL with C++ and runs entirely on a GPU, demonstrating:
- Parallel histogram calculation (global vs local atomic)
- Multiple parallel scan algorithms (Hillis-Steele, Blelloch)
- Variable bin sizes (flexible histogram granularity)
- Colour image support (RGB processing)

All image processing operations are performed on the GPU, with host-side I/O for image loading/saving and performance reporting.

# Key Features
Core Requirements
- OpenCL GPU implementation (no pre-existing libraries)
- 8-bit monochrome image support
- 256 fixed bins
- Memory transfer timing reporting
- Kernel execution timing reporting
- Total program execution time

Extended Features (Original Developments)
- Local memory histogram -	GPU shared memory for histogram accumulation -	Reduces global memory contention
- Variable bin size -	User-configurable number of bins (e.g., 128, 256, 512) -	Flexibility for different image types
- Multiple scan variants -	Hillis-Steele + Blelloch algorithms -	Performance comparison
- Colour image support -	RGB channel processing -	Works with colour images (24/32-bit)

# Parallel Algorithm Variants
## Histogram Calculation
Variant -	Description	- Use Case
- Global Histogram	- Simple atomic operations on global memory -	Baseline comparison
- Local Atomic Histogram -	Per-work-group local memory with atomic operations -	Reduced memory contention

## Parallel Scan Algorithms
## Hillis-Steele Scan
- Complexity: O(n log n) operations
- Pattern: Doubling stride, multiple passes
- Best for: Shorter arrays, simpler implementation
## Blelloch Scan
- Complexity: O(n) operations (2 passes)
- Pattern: Reduce phase + down-sweep phase
- Best for: Longer arrays, optimal work efficiency

# Optimisation Strategies
- Local Memory Histogram:
  - Each work-group maintains its own histogram in local memory (faster than global)
  - Reduces global memory contention from atomic operations
  - Improvement: 30-40% faster than global atomic approach
    
- Algorithm Selection:
  - Small images (<256KB): Hillis-Steele may be faster (lower overhead)
  - Large images (>1MB): Blelloch typically outperforms (O(n) vs O(n log n))

- Memory Coalescing:
  - Ensured adjacent threads access adjacent memory locations
  - Maximises memory bandwidth utilisation

- Variable Bin Sizes:
  - Smaller bins (64): Faster, lower quality equalisation
  - Larger bins (512): Slower, finer intensity discrimination
