from setuptools import setup
from torch.utils import cpp_extension

setup(name="quad_iou",
      ext_modules=[cpp_extension.CUDAExtension('quad_iou', ['quad_iou_tensors_cuda.cpp', 'quad_iou_tensors.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension}
      )
