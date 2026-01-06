from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import torch

sources = ['src/fastnorm.cpp', 'src/cpu/rmsnorm.cpp']
macros = []

if torch.cuda.is_available():
    sources.append('src/cuda/rmsnorm.cu')
    macros.append(('WITH_CUDA', None))
    ext = CUDAExtension(
        name='fastnorm',
        sources=sources,
        include_dirs=['include'],
        define_macros=macros
    )
else:
    ext = CppExtension(
        name='fastnorm',
        sources=sources,
        include_dirs=['include'],
    )

setup(
    name='fastnorm_cpp',
    version='0.1.3',
    ext_modules=[ext],
    cmdclass={'build_ext': BuildExtension}
)