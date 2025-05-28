#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <cuda_bf16.h>
#include <chrono>

/*
 * PyTorch CUDA Extension for Stochastic Copy, Stochastic BF16 Rounding, and Fused Optimizer
 *
 * This file provides C++ wrappers and PyBind11 bindings for CUDA kernels that perform:
 *   - Stochastic copy of float32 and bfloat16 tensors with random noise
 *   - Stochastic BF16 rounding with configurable probability and magnitude
 *   - Fused optimizer kernel for parameter, EMA, and EMA2 updates with stochastic rounding
 *
 * These functions are designed to be called from Python via the torch extension mechanism.
 *
 * Author: [Your Name]
 * Date: [Date]
 */

// Declare the launchers from the .cu file
void copy_stochastic_cuda_launcher(
    float* target, const float* source, int64_t numel, uint64_t seed, cudaStream_t stream);
void copy_stochastic_bf16_cuda_launcher(
    __nv_bfloat16* target, const float* source, int64_t numel, uint64_t seed, cudaStream_t stream);
void stochastic_bf16_rounding_launcher(
    __nv_bfloat16* target, const float* source, int64_t numel, float probability, float magnitude, uint64_t seed, cudaStream_t stream);

// Declare the fused optimizer launcher
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
    cudaStream_t stream);

/**
 * @brief C++ wrapper for launching the CUDA kernel for stochastic copy of float32 tensors.
 *
 * @param target Output float32 tensor (PyTorch CUDA tensor).
 * @param source Input float32 tensor (PyTorch CUDA tensor).
 * @param seed Random seed for noise generation.
 */
void copy_stochastic_cuda(
    at::Tensor target, at::Tensor source, uint64_t seed)
{
    //TORCH_CHECK(target.is_cuda(), "target must be a CUDA tensor");
    //TORCH_CHECK(source.is_cuda(), "source must be a CUDA tensor");
    //TORCH_CHECK(target.numel() == source.numel(), "Tensors must have same number of elements");
    //TORCH_CHECK(target.scalar_type() == at::kFloat, "target must be float32");
    //TORCH_CHECK(source.scalar_type() == at::kFloat, "source must be float32");

    copy_stochastic_cuda_launcher(
        target.data_ptr<float>(),
        source.data_ptr<float>(),
        target.numel(),
        seed,
        at::cuda::getCurrentCUDAStream()
    );
}

/**
 * @brief C++ wrapper for launching the CUDA kernel for stochastic copy from float32 to bfloat16.
 *
 * @param target Output bfloat16 tensor (PyTorch CUDA tensor).
 * @param source Input float32 tensor (PyTorch CUDA tensor).
 * @param seed Random seed for noise generation.
 */
void copy_stochastic_bf16_cuda(
    at::Tensor target, at::Tensor source, uint64_t seed)
{
    //TORCH_CHECK(target.is_cuda(), "target must be a CUDA tensor");
    //TORCH_CHECK(source.is_cuda(), "source must be a CUDA tensor");
    //TORCH_CHECK(target.numel() == source.numel(), "Tensors must have same number of elements");
    //TORCH_CHECK(target.scalar_type() == at::kBFloat16, "target must be bfloat16");
    //TORCH_CHECK(source.scalar_type() == at::kFloat, "source must be float32");

    copy_stochastic_bf16_cuda_launcher(
        reinterpret_cast<__nv_bfloat16*>(target.data_ptr<at::BFloat16>()),
        source.data_ptr<float>(),
        target.numel(),
        seed,
        at::cuda::getCurrentCUDAStream()
    );
}

/**
 * @brief C++ wrapper for launching the fused optimizer CUDA kernel for parameter, EMA, and EMA2 update.
 *
 * @param param Parameter tensor (bfloat16, PyTorch CUDA tensor).
 * @param ema Exponential moving average tensor (bfloat16, PyTorch CUDA tensor).
 * @param ema2 Exponential moving average of squared gradients (bfloat16, PyTorch CUDA tensor).
 * @param grad Gradient tensor (float32, PyTorch CUDA tensor).
 * @param lr Learning rate.
 * @param ema_beta Decay rate for EMA.
 * @param ema2_beta Decay rate for EMA2.
 */
void fused_optimizer(
    at::Tensor param,
    at::Tensor ema,
    at::Tensor ema2,
    at::Tensor grad,
    float lr,
    float ema_beta,
    float ema2_beta)
{
    //TORCH_CHECK(param.is_cuda() && ema.is_cuda() && ema2.is_cuda() && grad.is_cuda(), "All tensors must be CUDA tensors");
    //TORCH_CHECK(param.scalar_type() == at::kBFloat16, "param must be bfloat16");
    //TORCH_CHECK(ema.scalar_type() == at::kBFloat16, "ema must be bfloat16");
    //TORCH_CHECK(ema2.scalar_type() == at::kBFloat16, "ema2 must be bfloat16");
    //TORCH_CHECK(grad.scalar_type() == at::kFloat, "grad must be float32");
    //TORCH_CHECK(param.numel() == grad.numel() && ema.numel() == grad.numel() && ema2.numel() == grad.numel(), "All tensors must have the same number of elements");

    uint64_t seed = std::chrono::system_clock::now().time_since_epoch().count();

    fused_optimizer_kernel_launcher(
        reinterpret_cast<__nv_bfloat16*>(param.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(ema.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(ema2.data_ptr<at::BFloat16>()),
        grad.data_ptr<float>(),
        grad.numel(),
        lr,
        ema_beta,
        ema2_beta,
        seed,
        at::cuda::getCurrentCUDAStream()
    );
}

/**
 * @brief Overload for copy_stochastic_cuda with automatic seed generation.
 *
 * @param target Output float32 tensor (PyTorch CUDA tensor).
 * @param source Input float32 tensor (PyTorch CUDA tensor).
 */
void copy_stochastic_cuda(
    at::Tensor target, at::Tensor source)
{
    uint64_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    copy_stochastic_cuda(target, source, seed);
}

/**
 * @brief Overload for copy_stochastic_bf16_cuda with automatic seed generation.
 *
 * @param target Output bfloat16 tensor (PyTorch CUDA tensor).
 * @param source Input float32 tensor (PyTorch CUDA tensor).
 */
void copy_stochastic_bf16_cuda(
    at::Tensor target, at::Tensor source)
{
    uint64_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    copy_stochastic_bf16_cuda(target, source, seed);
}

/**
 * @brief C++ wrapper for launching the CUDA kernel for stochastic BF16 rounding with probability and magnitude.
 *
 * @param target Output bfloat16 tensor (PyTorch CUDA tensor).
 * @param source Input float32 tensor (PyTorch CUDA tensor).
 * @param probability Probability of applying noise to each element.
 * @param magnitude Magnitude of noise to apply.
 * @param seed Random seed for noise generation.
 */
void stochastic_bf16_rounding_cuda(
    at::Tensor target, at::Tensor source, float probability, float magnitude, uint64_t seed)
{
    stochastic_bf16_rounding_launcher(
        reinterpret_cast<__nv_bfloat16*>(target.data_ptr<at::BFloat16>()),
        source.data_ptr<float>(),
        target.numel(),
        probability,
        magnitude,
        seed,
        at::cuda::getCurrentCUDAStream()
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("copy_stochastic_cuda", (void (*)(at::Tensor, at::Tensor, uint64_t)) &copy_stochastic_cuda, "Stochastic copy (CUDA) with seed");
    m.def("copy_stochastic_bf16_cuda", (void (*)(at::Tensor, at::Tensor, uint64_t)) &copy_stochastic_bf16_cuda, "Stochastic copy to bfloat16 (CUDA) with seed");
    m.def("fused_optimizer", &fused_optimizer, "Fused optimizer kernel (CUDA)");
    m.def("fused_optimizer", [](at::Tensor param, at::Tensor ema, at::Tensor ema2, at::Tensor grad, float lr, float ema_beta, float ema2_beta, uint64_t seed) {
        //TORCH_CHECK(param.is_cuda() && ema.is_cuda() && ema2.is_cuda() && grad.is_cuda(), "All tensors must be CUDA tensors");
        //TORCH_CHECK(param.scalar_type() == at::kBFloat16, "param must be bfloat16");
        //TORCH_CHECK(ema.scalar_type() == at::kBFloat16, "ema must be bfloat16");
        //TORCH_CHECK(ema2.scalar_type() == at::kBFloat16, "ema2 must be bfloat16");
        //TORCH_CHECK(grad.scalar_type() == at::kFloat, "grad must be float32");
        //TORCH_CHECK(param.numel() == grad.numel() && ema.numel() == grad.numel() && ema2.numel() == grad.numel(), "All tensors must have the same number of elements");
        fused_optimizer_kernel_launcher(
            reinterpret_cast<__nv_bfloat16*>(param.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(ema.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(ema2.data_ptr<at::BFloat16>()),
            grad.data_ptr<float>(),
            grad.numel(),
            lr,
            ema_beta,
            ema2_beta,
            seed,
            at::cuda::getCurrentCUDAStream()
        );
    }, "Fused optimizer kernel (CUDA) with seed");
    // Register new kernel
    m.def("stochastic_bf16_rounding_cuda", &stochastic_bf16_rounding_cuda, "Stochastic BF16 rounding (CUDA) with probability and magnitude");
} 