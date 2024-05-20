# IoU(Intersection over Union) Calculation for Quadrilaterals(Torch extension)

Torch extension for calculating IoU (Intersection over Union) for quadrilaterals. It can calculate IoU either for a 1-to-1 match or an M-to-N match, returning an IoU matrix with M rows and N columns. Torch CUDA/CPP extensions are used for binding the code to Torch.

## Example usage:
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
