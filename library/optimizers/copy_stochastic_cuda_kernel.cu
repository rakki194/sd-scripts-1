#include <cuda.h>
#include <curand_kernel.h>
#include <ATen/cuda/CUDAContext.h>

__device__ uint32_t xorshift32(uint32_t x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

__global__ void copy_stochastic_cuda_kernel_vec4(
    float* __restrict__ target, const float* __restrict__ source, int64_t numel, uint64_t seed)
{
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int tidx = idx + i;
        if (tidx < numel) {
            uint32_t rand_state = static_cast<uint32_t>(seed) ^ static_cast<uint32_t>(tidx);
            uint32_t rand16 = xorshift32(rand_state) & 0xFFFF;
            int32_t src_bits = __float_as_int(source[tidx]);
            int32_t result = src_bits + rand16;
            result &= 0xFFFF0000;
            target[tidx] = __int_as_float(result);
        }
    }
}

// C++-style launcher
void copy_stochastic_cuda_launcher(
    float* target, const float* source, int64_t numel, uint64_t seed, cudaStream_t stream)
{
    int threads = 1024;
    int vec = 4;
    int blocks = (numel + threads * vec - 1) / (threads * vec);
    copy_stochastic_cuda_kernel_vec4<<<blocks, threads, 0, stream>>>(
        target, source, numel, seed
    );
} 