#ifndef QUAD_IOU
#define QUAD_IOU
#include <torch/torch.h>
#include <iostream>

torch::Tensor calculateIoUCudaTorch(torch::Tensor quad_0, torch::Tensor quad_1, bool sort_input_quads);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Calculate the Intersection over Union (IoU) of two quadrilaterals using GPU acceleration.";
        
    m.def("calculateIoU", &calculateIoUCudaTorch, 
    "Parameters:\n"
    "    quad_0 (torch.Tensor): The first quadrilateral(s), represented as a tensor of shape [N, 4, 2].\n"
    "    quad_1 (torch.Tensor): The second quadrilateral(s), represented as a tensor of shape [M, 4, 2].\n"
    "    sort_input_quads (bool, optional): If True, sort the vertices of the quads before IoU calculation. Defaults to False.\n\n"
    "Returns:\n"
    "    torch.Tensor: A matrix containing IoU values where i,j value is iou between i-th quadrilateral from 1st input tensor and j-th quadrilateral from 2nd input tensor.\n\n"
    "Examples:\n"
    "    >>> a = torch.rand((200, 4, 2)).cuda()\n"
    "    >>> b = torch.rand((300, 4, 2)).cuda()\n"
    "    >>> iou_matrix = quad_iou.calculateIoU(a, b, True) # returns tensor of shape [200, 300]\n",
    py::arg("quad0"), py::arg("quad1"), py::arg("sort_input_quads"));
};

#endif