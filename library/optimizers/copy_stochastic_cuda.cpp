#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <chrono>

// Declare the launcher from the .cu file
void copy_stochastic_cuda_launcher(
    float* target, const float* source, int64_t numel, uint64_t seed, cudaStream_t stream);

void copy_stochastic_cuda(
    at::Tensor target, at::Tensor source)
{
    TORCH_CHECK(target.is_cuda(), "target must be a CUDA tensor");
    TORCH_CHECK(source.is_cuda(), "source must be a CUDA tensor");
    TORCH_CHECK(target.numel() == source.numel(), "Tensors must have same number of elements");
    TORCH_CHECK(target.scalar_type() == at::kFloat, "target must be float32");
    TORCH_CHECK(source.scalar_type() == at::kFloat, "source must be float32");

    // Use a simple time-based seed
    uint64_t seed = std::chrono::system_clock::now().time_since_epoch().count();

    copy_stochastic_cuda_launcher(
        target.data_ptr<float>(),
        source.data_ptr<float>(),
        target.numel(),
        seed,
        at::cuda::getCurrentCUDAStream()
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("copy_stochastic_cuda", &copy_stochastic_cuda, "Stochastic copy (CUDA)");
} 