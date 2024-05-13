#ifndef QUAD_IOU
#define QUAD_IOU
#include <torch/torch.h>
#include <iostream>

torch::Tensor calculateIoUCudaTorch(torch::Tensor quad_0, torch::Tensor quad_1);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("calculateIoU", &calculateIoUCudaTorch, "IoU Calculation for Quads (CUDA)");
};

#endif