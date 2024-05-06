#ifndef QUAD_IOU
#define QUAD_IOU
#include <torch/torch.h>
#include <iostream>

torch::Tensor calculateIoUCudaTorch(torch::Tensor quad_0, torch::Tensor quad_1);
torch::Tensor calculateIoU(torch::Tensor quads_0, torch::Tensor quads_1){
    return calculateIoUCudaTorch(quads_0, quads_1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("calculateIoU", &calculateIoU, "IoU Calculation for Quads (CUDA)");
};

#endif