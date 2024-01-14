#ifndef QUAD_IOU
#define QUAD_IOU
#include <torch/torch.h>
#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor");
#define CHECK_INPUT(x) CHECK_CUDA(x);


torch::Tensor calculateIoUCudaTorch(torch::Tensor quad_0, torch::Tensor quad_1);
torch::Tensor calculateIoU(torch::Tensor quads_0, torch::Tensor quads_1){
    CHECK_INPUT(quads_0);
    CHECK_INPUT(quads_1);
    return calculateIoUCudaTorch(quads_0, quads_1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("calculateIoU", &calculateIoU, "IoU Calculation for Quads (CUDA)");
};

#endif