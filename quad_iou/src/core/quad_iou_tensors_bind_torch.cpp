#ifndef QUAD_IOU
#define QUAD_IOU
#include <torch/torch.h>
#include <iostream>

torch::Tensor calculateIoUCudaTorch(torch::Tensor quad_0, torch::Tensor quad_1, bool sort_input_quads);
torch::Tensor calculateIoUCPUTorch(torch::Tensor quad_0, torch::Tensor quad_1, bool sort_input_quads);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("calculateIoUCuda", &calculateIoUCudaTorch);
    m.def("calculateIoUCPU", &calculateIoUCPUTorch);
};

#endif