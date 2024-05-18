# IoU(Intersection over Union) Calculation for Quadrilaterals(CUDA Kernel)

Cuda kernel for calculating IoU for quadrilaterals. It can calculate IoU either for 1->1 match or N->M match, returning an iou matrix with N rows and M columns. Torch CUDA extensions are used for running the compiled kernels. 


## Installation

**NOTE:** Installation and usage of this module requires NVIDIA GPU, gcc, nvcc and torch installed.

1. Run `pip install quad_iou`
2. To confirm installation, run `python tryout_scripts/usage.py` to test it out. Expected output is `0.25`

## Example usage:
```
import torch
import quad_iou

# NxM quadrilaterals
a = torch.rand((200, 4, 2)).cuda()
b = torch.rand((300, 4, 2)).cuda()

# sort_input_quads indicate whether kernel should sort the quadrilateral corners
# clockwise before calculating iou
iou_matrix = quad_iou.calculate_iou(a, b, sort_input_quads=True) # returns tensor of shape [200, 300]

# 1x1 case
a = torch.tensor([0.0, 0, 300, 0, 300, 300, 0, 300]).cuda()
b = torch.tensor([0.0, 0, 150, 0, 150, 150, 0, 150]).cuda()
# Module expects tensor of shape [N, 4, 2], so we reshape the tensors
a = a.reshape(-1, 4, 2)
b = b.reshape(-1, 4, 2)
iou = quad_iou.calculate_iou(a, b, sort_input_quads=True)
```

## Comparison with Shapely library

While CPUs and GPUs are not to be compared for speed, we provide a script to demonstrate the potential speedup when using a GPU. To compare the execution time of calculating the IoU for quadrilaterals using the `Shapely` library versus our GPU-accelerated implementation, run `python tryout_scripts/comparison.py`


## TODO
- [ ] Add more tests in `tests/test.py` for dealing with MxN quadrilaterals, now tests are only for 1->1 quadrilaterals
- [ ] Add more tests in `tests/test.py` for dealing with 1x1 quadrilateral without sorting `sort_input_quads=False`
- [x] Make package available on pypi