#include <cuda.h>
#include <curand_kernel.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

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

// bfloat16 stochastic rounding kernel (bit-manipulation, matches Python)
__global__ void copy_stochastic_bf16_cuda_kernel(
    __nv_bfloat16* __restrict__ target, const float* __restrict__ source, int64_t numel, uint64_t seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float src = source[idx];
        int32_t src_bits = __float_as_int(src);
        if (isnan(src)) {
            target[idx] = __float2bfloat16(NAN);
        } else if (isinf(src)) {
            target[idx] = __float2bfloat16(src);
        } else {
            uint32_t rand_state = static_cast<uint32_t>(seed) ^ static_cast<uint32_t>(idx);
            uint32_t rand16 = xorshift32(rand_state) & 0xFFFF;
            int32_t result = src_bits + rand16;
            result &= 0xFFFF0000;
            float rounded = __int_as_float(result);
            target[idx] = __float2bfloat16(rounded);
        }
    }
}

// C++-style launcher for float32
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

// C++-style launcher for bfloat16
void copy_stochastic_bf16_cuda_launcher(
    __nv_bfloat16* target, const float* source, int64_t numel, uint64_t seed, cudaStream_t stream)
{
    int threads = 1024;
    int blocks = (numel + threads - 1) / threads;
    copy_stochastic_bf16_cuda_kernel<<<blocks, threads, 0, stream>>>(
        target, source, numel, seed
    );
} 