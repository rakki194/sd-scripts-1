#include <cuda.h>
#include <curand_kernel.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void copy_stochastic_cuda_kernel(
    float* target, const float* source, int64_t numel, uint64_t seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    // Setup per-thread RNG
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    // Reinterpret float as int32
    int32_t src_bits = reinterpret_cast<const int32_t*>(source)[idx];

    // Generate random 16-bit integer
    int32_t rand16 = curand(&state) & 0xFFFF;

    // Add random to lower 16 bits
    int32_t result = src_bits + rand16;

    // Mask off lower 16 bits
    result &= 0xFFFF0000;

    // Reinterpret as float
    float out = *reinterpret_cast<float*>(&result);

    target[idx] = out;
}

// C++-style launcher
void copy_stochastic_cuda_launcher(
    float* target, const float* source, int64_t numel, uint64_t seed, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    copy_stochastic_cuda_kernel<<<blocks, threads, 0, stream>>>(
        target, source, numel, seed
    );
} 