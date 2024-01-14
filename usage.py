import torch
import quad_iou

a = torch.tensor([0.0, 0, 300, 0, 300, 300, 0, 300]).cuda()
b = torch.tensor([0.0, 0, 150, 0, 150, 150, 0, 150]).cuda()
a = a.reshape(-1, 4, 2)
b = b.reshape(-1, 4, 2)

iou = quad_iou.calculateIoU(a, b)
print(iou)