import torch
import quad_iou

# NxM quadrilaterals
a = torch.rand((200, 4, 2)).cuda()
b = torch.rand((300, 4, 2)).cuda()
iou_matrix = quad_iou.calculateIoU(a, b) # returns tensor of shape [200, 300]
 
# 1x1 case
a = torch.tensor([0.0, 0, 300, 0, 300, 300, 0, 300]).cuda()
b = torch.tensor([0.0, 0, 150, 0, 150, 150, 0, 150]).cuda()
# Module expects tensor of shape [N, 4, 2], so we reshape the tensors
a = a.reshape(-1, 4, 2)
b = b.reshape(-1, 4, 2)
iou = quad_iou.calculateIoU(a, b)
print(iou)