# IoU(Intersection over Union) Calculation for Quadrilaterals(Torch extension)

Torch extension for calculating IoU (Intersection over Union) for quadrilaterals. It can calculate IoU either for a 1-to-1 match or an M-to-N match, returning an IoU matrix with M rows and N columns. Torch CUDA/CPP extensions are used for binding the code to Torch.

## Installation

**NOTE:** Installation and usage of this package requires gcc and torch installed. If you'd like to use cuda version NVIDIA GPU, gcc, nvcc and torch installation is mandatory.

1. Run `pip install quad_iou`
2. To confirm installation, run `python tryout_scripts/usage_cpu.py` to test it out. Expected output is `0.25`

## Example usage
```
import torch
import quad_iou

# NxM quadrilaterals
a = torch.rand((200, 4, 2)).cuda() # Can also be without cuda
b = torch.rand((300, 4, 2)).cuda()

# sort_input_quads indicate whether kernel should sort the quadrilateral corners
# clockwise before calculating iou
iou_matrix = quad_iou.calculate_iou(a, b, sort_input_quads=True) # returns tensor of shape [200, 300]

# 1x1 case
a = torch.tensor([0.0, 0, 300, 0, 300, 300, 0, 300]).cuda() # Can also be without cuda
b = torch.tensor([0.0, 0, 150, 0, 150, 150, 0, 150]).cuda()
# Module expects tensor of shape [N, 4, 2], so we reshape the tensors
a = a.reshape(-1, 4, 2)
b = b.reshape(-1, 4, 2)
iou = quad_iou.calculate_iou(a, b, sort_input_quads=True)
```

## Example usage with Colab
You can try the package on Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Po7oJywlEXEeMJZRqxVNH3jEpRUfA2d_?usp=sharing)

## Comparison with shapely

Shapely is one of the libraries that can be used for IoU (Intersection over Union) calculations for quadrilaterals in Python. To evaluate the performance, we compare our implementation with Shapely's, attempting to utilize list comprehensions as much as possible to avoid slowing down the Shapely code. The results of this comparison are shown below.

![Comparison][comparison/iou_benchmark_plot.png]

## TODO
- [ ] Add more tests in `tests/test.py` for dealing with MxN quadrilaterals, now tests are only for 1->1 quadrilaterals
- [ ] Add more tests in `tests/test.py` for dealing with 1x1 quadrilateral without sorting `sort_input_quads=False`
- [x] Make package available on pypi
- [x] CPU version
- [ ] MPS version
- [x] Parallelize on CPU(Pragma ignored during compilation)

