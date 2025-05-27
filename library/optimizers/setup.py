from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='copy_stochastic_cuda',
    ext_modules=[
        CUDAExtension(
            'copy_stochastic_cuda',
            [
                'copy_stochastic_cuda.cpp',
                'copy_stochastic_cuda_kernel.cu',
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
) 