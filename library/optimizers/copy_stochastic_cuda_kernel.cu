#include <cuda.h>
#include <curand_kernel.h>
#include <ATen/cuda/CUDAContext.h>

__device__ uint32_t xorshift32(uint32_t x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

__global__ void copy_stochastic_cuda_kernel_opt(
    float* __restrict__ target, const float* __restrict__ source, int64_t numel, uint64_t seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    // Fast stateless RNG per thread
    uint32_t rand_state = static_cast<uint32_t>(seed) ^ static_cast<uint32_t>(idx);
    uint32_t rand16 = xorshift32(rand_state) & 0xFFFF;

    int32_t src_bits = __float_as_int(source[idx]);
    int32_t result = src_bits + rand16;
    result &= 0xFFFF0000;
    target[idx] = __int_as_float(result);
}

// C++-style launcher
void copy_stochastic_cuda_launcher(
    float* target, const float* source, int64_t numel, uint64_t seed, cudaStream_t stream)
{
    int threads = 1024;
    int blocks = (numel + threads - 1) / threads;
    copy_stochastic_cuda_kernel_opt<<<blocks, threads, 0, stream>>>(
        target, source, numel, seed
    );
} 