#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// Kernel declaration
void copy_stochastic_cuda_kernel(
    float* target, const float* source, int64_t numel, uint64_t seed);

void copy_stochastic_cuda_launcher(
    at::Tensor& target, const at::Tensor& source, uint64_t seed)
{
    const int threads = 256;
    const int64_t numel = source.numel();
    const int blocks = (numel + threads - 1) / threads;

    // Only support float32
    copy_stochastic_cuda_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        target.data_ptr<float>(),
        source.data_ptr<float>(),
        numel,
        seed
    );
}

void copy_stochastic_cuda(
    at::Tensor& target, const at::Tensor& source)
{
    TORCH_CHECK(target.is_cuda(), "target must be a CUDA tensor");
    TORCH_CHECK(source.is_cuda(), "source must be a CUDA tensor");
    TORCH_CHECK(target.numel() == source.numel(), "Tensors must have same number of elements");
    TORCH_CHECK(target.scalar_type() == at::kFloat, "target must be float32");
    TORCH_CHECK(source.scalar_type() == at::kFloat, "source must be float32");

    uint64_t seed = at::cuda::getDefaultCUDAGenerator().current_seed();
    copy_stochastic_cuda_launcher(target, source, seed);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("copy_stochastic_cuda", &copy_stochastic_cuda, "Stochastic copy (CUDA)");
} 