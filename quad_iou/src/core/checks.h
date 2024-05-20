#include <torch/extension.h>

namespace checks {
    void check_tensor_validity(torch::Tensor& quad_0, torch::Tensor& quad_1){
        TORCH_CHECK(quad_0.is_contiguous() && quad_1.is_contiguous(), "Input tensors must be contiguous");
        TORCH_CHECK(quad_0.numel() > 0 && quad_1.numel() > 0, "Input tensors must not empty");
        TORCH_CHECK(quad_0.dim() == 3, "Input tensor must contain 3 dimensions");
        TORCH_CHECK(quad_0.size(1) == 4 && quad_1.size(1) == 4, "Input tensors must have 4 values on 2nd dim(dim=1)");
        TORCH_CHECK(quad_0.size(2) == 2 && quad_1.size(2) == 2, "Input tensors must have 2 values on last dim(dim=2)");
    }
}