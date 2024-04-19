import numpy as np
import torch
import quad_iou
from shapely import Polygon

def generate_convex_quadrilateral():
    # Generate four random points
    points = np.random.rand(4, 2)

    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)

    # Calculate the angles of the points relative to the centroid
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])

    # Sort the points by angle
    order = angles.argsort()
    points = points[order]

    # Return the sorted points
    return points * 100

def is_convex(quad: Polygon):
    convex_hull = quad.convex_hull
    return quad.equals(convex_hull)

def calc_iou_poly(quad_0: Polygon, quad_1: Polygon):
    intersect_area = quad_0.intersection(quad_1).area
    union_area = quad_0.area + quad_1.area - intersect_area
    return intersect_area / union_area

def main():
    error_quads = []
    not_errors = []

    for _ in range(100_000):
        quad_1 = generate_convex_quadrilateral()
        quad_2 = generate_convex_quadrilateral()
        quad_poly_1 = Polygon(quad_1)
        quad_poly_2 = Polygon(quad_2)
        quad_1_tensor = torch.tensor(quad_1).unsqueeze(0).cuda()
        quad_2_tensor = torch.tensor(quad_2).unsqueeze(0).cuda()
        if is_convex(quad_poly_1) and is_convex(quad_poly_2):
            iou_shapely = calc_iou_poly(quad_poly_1, quad_poly_2)
            iou_tensor = quad_iou.calculateIoU(quad_1_tensor, quad_2_tensor).item()
            if abs(iou_shapely - iou_tensor) > 0.0001:
                print("ERROR")
                error_quads.append((quad_1, quad_2))
            else:
                not_errors.append(iou_tensor)

    print(error_quads)

if __name__ == "__main__":
    main()
