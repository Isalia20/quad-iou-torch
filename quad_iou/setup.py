from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import torch

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

ext_modules = [
    CppExtension("quad_iou_cpu", [
        "src/core/quad_iou_tensors_bind_cpu.cpp",
        "src/core/quad_iou_tensors.cpp",
    ],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp']
    )
]

if torch.cuda.is_available():
    ext_modules.append(
        CUDAExtension('quad_iou_cuda', [
            'src/core/quad_iou_tensors_bind_cuda.cpp',
            'src/core/quad_iou_tensors.cu',
        ])
    )

setup(
    name="quad_iou",
    version="0.1.0",
    author="Irakli Salia",
    author_email="irakli.salia854@gmail.com",
    description="Torch extension for calculating IoU(Intersection over Union) for quadrilaterals(MxN)",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/Isalia20/quad-iou-torch",
    package_dir={'': 'src'},
    packages=find_packages(where="src"),
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires='>=3.9',
)