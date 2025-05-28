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

// Optimized bfloat16 stochastic rounding kernel
__global__ void copy_stochastic_bf16_cuda_kernel(
    __nv_bfloat16* __restrict__ target, const float* __restrict__ source, int64_t numel, uint64_t seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float src = __ldg(&source[idx]);
        int32_t src_bits = __float_as_int(src);
        if (isfinite(src)) {
            uint32_t rand_state = static_cast<uint32_t>(seed) ^ static_cast<uint32_t>(idx);
            uint32_t rand16 = xorshift32(rand_state) & 0xFFFF;
            int32_t result = src_bits + rand16;
            result &= 0xFFFF0000;
            float rounded = __int_as_float(result);
            target[idx] = __float2bfloat16(rounded);
        } else if (isnan(src)) {
            target[idx] = __float2bfloat16(NAN);
        } else {
            target[idx] = __float2bfloat16(src);
        }
    }
}

// C++-style launcher for float32
void copy_stochastic_cuda_launcher(
    float* target, const float* source, int64_t numel, uint64_t seed, cudaStream_t stream)
{
    int threads = 512;
    int vec = 4;
    int blocks = (numel + threads * vec - 1) / (threads * vec);
    copy_stochastic_cuda_kernel_vec4<<<blocks, threads, 0, stream>>>(
        target, source, numel, seed
    );
}

// C++-style launcher for bfloat16 (optimized)
void copy_stochastic_bf16_cuda_launcher(
    __nv_bfloat16* target, const float* source, int64_t numel, uint64_t seed, cudaStream_t stream)
{
    int threads = 512; // Try 256, 512, 1024 for best performance on your GPU
    int blocks = (numel + threads - 1) / threads;
    copy_stochastic_bf16_cuda_kernel<<<blocks, threads, 0, stream>>>(
        target, source, numel, seed
    );
}

__global__ void fused_optimizer_kernel(
    __nv_bfloat16* __restrict__ param,
    __nv_bfloat16* __restrict__ ema,
    __nv_bfloat16* __restrict__ ema2,
    const float* __restrict__ grad,
    int64_t numel,
    float lr,
    float ema_beta,
    float ema2_beta,
    uint64_t seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        // param update
        float p = __bfloat162float(param[idx]);
        float g = __ldg(&grad[idx]);
        float new_p = p - lr * g;
        int32_t src_bits = __float_as_int(new_p);
        uint32_t rand_state = static_cast<uint32_t>(seed) ^ static_cast<uint32_t>(idx);
        uint32_t rand16 = xorshift32(rand_state) & 0xFFFF;
        int32_t result = src_bits + rand16;
        result &= 0xFFFF0000;
        float rounded_p = __int_as_float(result);
        param[idx] = __float2bfloat16(rounded_p);

        // ema update
        float e = __bfloat162float(ema[idx]);
        float new_e = ema_beta * e + (1.0f - ema_beta) * rounded_p;
        ema[idx] = __float2bfloat16(new_e);

        // ema2 update
        float e2 = __bfloat162float(ema2[idx]);
        float new_e2 = ema2_beta * e2 + (1.0f - ema2_beta) * (g * g);
        ema2[idx] = __float2bfloat16(new_e2);
    }
}

void fused_optimizer_kernel_launcher(
    __nv_bfloat16* param,
    __nv_bfloat16* ema,
    __nv_bfloat16* ema2,
    const float* grad,
    int64_t numel,
    float lr,
    float ema_beta,
    float ema2_beta,
    uint64_t seed,
    cudaStream_t stream)
{
    int threads = 512;
    int blocks = (numel + threads - 1) / threads;
    fused_optimizer_kernel<<<blocks, threads, 0, stream>>>(
        param, ema, ema2, grad, numel, lr, ema_beta, ema2_beta, seed
    );
}

// Stochastic BF16 rounding kernel (probability/magnitude, noise-based)
__global__ void stochastic_bf16_rounding_kernel(
    __nv_bfloat16* __restrict__ target,
    const float* __restrict__ source,
    int64_t numel,
    float probability,
    float magnitude,
    uint64_t seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float src = __ldg(&source[idx]);
        uint32_t rand_state = static_cast<uint32_t>(seed) ^ static_cast<uint32_t>(idx);
        // Uniform random float in [0,1)
        float rand_prob = (xorshift32(rand_state) & 0xFFFFFF) / float(0x1000000);
        // Random sign
        float sign = ((xorshift32(rand_state + 17) & 1) ? 1.0f : -1.0f);
        float noise = 0.0f;
        if (rand_prob < probability) {
            noise = sign * magnitude;
        }
        float noisy = src + noise;
        target[idx] = __float2bfloat16(noisy);
    }
}

void stochastic_bf16_rounding_launcher(
    __nv_bfloat16* target,
    const float* source,
    int64_t numel,
    float probability,
    float magnitude,
    uint64_t seed,
    cudaStream_t stream)
{
    int threads = 512;
    int blocks = (numel + threads - 1) / threads;
    stochastic_bf16_rounding_kernel<<<blocks, threads, 0, stream>>>(
        target, source, numel, probability, magnitude, seed
    );
} 