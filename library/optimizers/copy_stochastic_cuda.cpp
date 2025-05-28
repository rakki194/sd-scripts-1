#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <cuda_bf16.h>
#include <chrono>

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

void fused_optimizer_kernel_vec4_launcher(
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

// Declare new launchers
void normalize_gradient_cuda_launcher(float* x, int64_t numel, float alpha, float epsilon, float mean, float std, cudaStream_t stream);
void normalize_gradient_channel_cuda_launcher(float* x, int64_t N, int64_t C, float alpha, float epsilon, const float* means, const float* stds, cudaStream_t stream);
void global_permutation_cuda_launcher(float* x, int64_t numel, uint64_t seed, cudaStream_t stream);

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

// New: bfloat16 binding
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

// Original overloads for backward compatibility
void copy_stochastic_cuda(
    at::Tensor target, at::Tensor source)
{
    uint64_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    copy_stochastic_cuda(target, source, seed);
}

void copy_stochastic_bf16_cuda(
    at::Tensor target, at::Tensor source)
{
    uint64_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    copy_stochastic_bf16_cuda(target, source, seed);
}

// Add C++ wrapper for new kernel
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

void fused_optimizer_vec4(
    at::Tensor param,
    at::Tensor ema,
    at::Tensor ema2,
    at::Tensor grad,
    float lr,
    float ema_beta,
    float ema2_beta,
    uint64_t seed
) {
    //TORCH_CHECK(param.is_cuda() && ema.is_cuda() && ema2.is_cuda() && grad.is_cuda(), "All tensors must be CUDA tensors");
    //TORCH_CHECK(param.scalar_type() == at::kBFloat16, "param must be bfloat16");
    //TORCH_CHECK(ema.scalar_type() == at::kBFloat16, "ema must be bfloat16");
    //TORCH_CHECK(ema2.scalar_type() == at::kBFloat16, "ema2 must be bfloat16");
    //TORCH_CHECK(grad.scalar_type() == at::kFloat, "grad must be float32");
    //TORCH_CHECK(param.numel() == grad.numel() && ema.numel() == grad.numel() && ema2.numel() == grad.numel(), "All tensors must have the same number of elements");

    fused_optimizer_kernel_vec4_launcher(
        reinterpret_cast<__nv_bfloat16*>(param.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(ema.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(ema2.data_ptr<at::BFloat16>()),
        grad.data_ptr<float>(),
        param.numel(),
        lr,
        ema_beta,
        ema2_beta,
        seed,
        at::cuda::getCurrentCUDAStream()
    );
}

void normalize_gradient_cuda(at::Tensor x, bool use_channels, float alpha, float epsilon, uint64_t seed) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    if (!use_channels) {
        float mean = x.mean().item<float>();
        float std = x.std().item<float>() + epsilon;
        normalize_gradient_cuda_launcher(x.data_ptr<float>(), x.numel(), alpha, epsilon, mean, std, at::cuda::getCurrentCUDAStream());
    } else {
        // Assume 2D (N, C) for simplicity
        TORCH_CHECK(x.dim() == 2, "Channel-wise normalization only supports 2D tensors");
        int64_t N = x.size(0);
        int64_t C = x.size(1);
        auto means = x.mean(0).contiguous();
        auto stds = x.std(0).add(epsilon).contiguous();
        normalize_gradient_channel_cuda_launcher(x.data_ptr<float>(), N, C, alpha, epsilon, means.data_ptr<float>(), stds.data_ptr<float>(), at::cuda::getCurrentCUDAStream());
    }
}

void global_permutation_cuda(at::Tensor x, uint64_t seed) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    global_permutation_cuda_launcher(x.data_ptr<float>(), x.numel(), seed, at::cuda::getCurrentCUDAStream());
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
    m.def("fused_optimizer_vec4", &fused_optimizer_vec4, "Vectorized fused optimizer kernel (vec4)");
    m.def("normalize_gradient_cuda", &normalize_gradient_cuda, "Normalize gradient (CUDA)");
    m.def("global_permutation_cuda", &global_permutation_cuda, "Global permutation (CUDA)");
} 