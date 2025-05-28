#include <cuda.h>
#include <curand_kernel.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

/*
 * CUDA Kernels for Stochastic Copy, Stochastic BF16 Rounding, and Fused Optimizer
 *
 * This file implements CUDA kernels and their launchers for stochastic rounding
 * and parameter updates for deep learning optimizers. The kernels include:
 *   - Stochastic copy of float32 and bfloat16 tensors with random noise for improved training dynamics
 *   - Stochastic BF16 rounding with configurable probability and magnitude
 *   - Fused optimizer kernel for parameter, EMA, and EMA2 updates with stochastic rounding
 *
 * Kernels are designed for use with PyTorch custom extensions and are called from C++ wrappers.
 */

__device__ uint32_t xorshift32(uint32_t x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

/**
 * @brief Simple xorshift32 pseudo-random number generator for device-side randomness.
 * @param x Input seed value.
 * @return Pseudo-random 32-bit unsigned integer.
 */

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
            int32_t src_bits = __float_as_int(__ldg(&source[tidx]));
            int32_t result = src_bits + rand16;
            result &= 0xFFFF0000;
            target[tidx] = __int_as_float(result);
        }
    }
}

/**
 * @brief CUDA kernel for stochastic copy of float32 tensors using vectorized (vec4) access.
 * Adds random noise to the lower 16 bits of each float32 element for stochastic rounding.
 *
 * @param target Output float32 tensor (device pointer).
 * @param source Input float32 tensor (device pointer).
 * @param numel Number of elements to process.
 * @param seed Random seed for noise generation.
 */

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

/**
 * @brief CUDA kernel for stochastic copy from float32 to bfloat16 with random rounding.
 * Adds random noise to the lower 16 bits before conversion to bfloat16 for improved training dynamics.
 *
 * @param target Output bfloat16 tensor (device pointer).
 * @param source Input float32 tensor (device pointer).
 * @param numel Number of elements to process.
 * @param seed Random seed for noise generation.
 */

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

/**
 * @brief Host launcher for copy_stochastic_cuda_kernel_vec4.
 *
 * @param target Output float32 tensor (device pointer).
 * @param source Input float32 tensor (device pointer).
 * @param numel Number of elements to process.
 * @param seed Random seed for noise generation.
 * @param stream CUDA stream to launch the kernel on.
 */

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

/**
 * @brief Host launcher for copy_stochastic_bf16_cuda_kernel.
 *
 * @param target Output bfloat16 tensor (device pointer).
 * @param source Input float32 tensor (device pointer).
 * @param numel Number of elements to process.
 * @param seed Random seed for noise generation.
 * @param stream CUDA stream to launch the kernel on.
 */

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
    float inv_ema_beta = 1.0f - ema_beta;
    float inv_ema2_beta = 1.0f - ema2_beta;
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

        // ema update (fused multiply-add)
        float e = __bfloat162float(ema[idx]);
        float new_e = __fmaf_rn(inv_ema_beta, rounded_p, ema_beta * e);
        ema[idx] = __float2bfloat16(new_e);

        // ema2 update (fused multiply-add)
        float e2 = __bfloat162float(ema2[idx]);
        float new_e2 = __fmaf_rn(inv_ema2_beta, g * g, ema2_beta * e2);
        ema2[idx] = __float2bfloat16(new_e2);
    }
}

/**
 * @brief CUDA kernel for fused parameter, EMA, and EMA2 update with stochastic bfloat16 rounding.
 * Performs parameter update, EMA, and EMA2 update in a single kernel for efficiency.
 *
 * @param param Parameter tensor (bfloat16, device pointer).
 * @param ema Exponential moving average tensor (bfloat16, device pointer).
 * @param ema2 Exponential moving average of squared gradients (bfloat16, device pointer).
 * @param grad Gradient tensor (float32, device pointer).
 * @param numel Number of elements to process.
 * @param lr Learning rate.
 * @param ema_beta Decay rate for EMA.
 * @param ema2_beta Decay rate for EMA2.
 * @param seed Random seed for stochastic rounding.
 */

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

/**
 * @brief Host launcher for fused_optimizer_kernel.
 *
 * @param param Parameter tensor (bfloat16, device pointer).
 * @param ema Exponential moving average tensor (bfloat16, device pointer).
 * @param ema2 Exponential moving average of squared gradients (bfloat16, device pointer).
 * @param grad Gradient tensor (float32, device pointer).
 * @param numel Number of elements to process.
 * @param lr Learning rate.
 * @param ema_beta Decay rate for EMA.
 * @param ema2_beta Decay rate for EMA2.
 * @param seed Random seed for stochastic rounding.
 * @param stream CUDA stream to launch the kernel on.
 */

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
    const float kFloat24 = 16777216.0f; // 0x1000000
    if (idx < numel) {
        float src = __ldg(&source[idx]);
        uint32_t rand_state = static_cast<uint32_t>(seed) ^ static_cast<uint32_t>(idx);
        float rand_prob = (xorshift32(rand_state) & 0xFFFFFF) / kFloat24;
        float sign = ((xorshift32(rand_state + 17) & 1) ? 1.0f : -1.0f);
        // Branchless noise logic
        float noise = (rand_prob < probability) ? (sign * magnitude) : 0.0f;
        float noisy = src + noise;
        target[idx] = __float2bfloat16(noisy);
    }
}

/**
 * @brief CUDA kernel for stochastic BF16 rounding with configurable probability and magnitude.
 * Adds noise to float32 input before conversion to bfloat16, controlled by probability and magnitude.
 *
 * @param target Output bfloat16 tensor (device pointer).
 * @param source Input float32 tensor (device pointer).
 * @param numel Number of elements to process.
 * @param probability Probability of applying noise to each element.
 * @param magnitude Magnitude of noise to apply.
 * @param seed Random seed for noise generation.
 */

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

/**
 * @brief Host launcher for stochastic_bf16_rounding_kernel.
 *
 * @param target Output bfloat16 tensor (device pointer).
 * @param source Input float32 tensor (device pointer).
 * @param numel Number of elements to process.
 * @param probability Probability of applying noise to each element.
 * @param magnitude Magnitude of noise to apply.
 * @param seed Random seed for noise generation.
 * @param stream CUDA stream to launch the kernel on.
 */ 