# import torch
# import quad_iou

# def create_quadrilateral_tensor(points):
#     return torch.tensor(points, dtype=torch.float32).reshape(-1, 4, 2).cuda()

# # a = create_quadrilateral_tensor([[[0, 0], [1, 0], [1, 1], [0, 1]]])
# # b = create_quadrilateral_tensor([[[0, 0], [1, 0], [1, 1], [0, 1]]])
# # a = create_quadrilateral_tensor([[[0, 0], [1, 0], [1, 1], [0, 1]], [[1, 1], [2, 1], [2, 2], [1, 2]]])
# # b = create_quadrilateral_tensor([[[0, 0], [1, 0], [1, 1], [0, 1]], [[2, 2], [3, 2], [3, 3], [2, 3]]])
# a = create_quadrilateral_tensor([[[0, 0], [1, 0], [1, 1], [0, 1]]]).double()
# b = create_quadrilateral_tensor([[[2, 0], [3, 0], [3, 1], [2, 1]]]).double()
# # (
# #         create_quadrilateral_tensor([[[0, 0], [2, 0], [1, 1], [0, 1]]]),
# #         create_quadrilateral_tensor([[[1, 0], [3, 0], [3, 1], [2, 1]]]),
# #         torch.tensor([1/5])  # Example value; replace with actual calculation
# #     ),

# calculated_iou = quad_iou.calculateIoU(a, b)
# print(calculated_iou)
# # expected_iou = torch.tensor(1.0).cuda()
# # print(torch.allclose(calculated_iou, expected_iou, atol=1e-6), f"Calculated IoU: {calculated_iou}, Expected IoU: {expected_iou}")


import shapely

temp_a = [0, 0, 2, 0, 2.5, 3, 0.5, 2.5]
temp_b = [0.2, 1, 0.8, 0.5, 1.7, 2.4, 1.4, 3.3]

# Function to transform list into coordinate pairs
def list_to_coords(lst):
    return [(lst[i], lst[i + 1]) for i in range(0, len(lst), 2)]

def create_quadrilateral_tensor(points, dtype):
    """Create a tensor for quadrilateral points, change its data type, and move it to the GPU."""
    return torch.tensor(points, dtype=dtype).reshape(-1, 4, 2).cuda()

# Create polygons
a = shapely.Polygon(list_to_coords(temp_a))
b = shapely.Polygon(list_to_coords(temp_b))

iou = shapely.intersection(a, b).area / shapely.union(a, b).area
# print(iou)

import torch
import quad_iou

quad_0 = create_quadrilateral_tensor(temp_a, dtype=torch.float32)
quad_1 = create_quadrilateral_tensor(temp_b, dtype=torch.float32)
iou = quad_iou.calculateIoU(quad_0, quad_1)
print(iou)




# ALL POINTS 0 and i: 1 is: 1.055000
# ALL POINTS 1 and i: 1 is: 2.638750
# ALL POINTS 0 and i: 2 is: 0.200000
# ALL POINTS 1 and i: 2 is: 1.000000
# ALL POINTS 0 and i: 3 is: 0.200000
# ALL POINTS 1 and i: 3 is: 1.000000