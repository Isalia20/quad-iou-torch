from shapely import Polygon
import torch
import quad_iou
import numpy as np
import time
from shapely import Polygon

def generate_quadrilateral():
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

def generate_convex_quads(num_quads_to_generate: int):
    quads = []
    while len(quads) != num_quads_to_generate:
        quad = generate_quadrilateral()
        quad_poly_1 = Polygon(quad)
        while not is_convex(quad_poly_1):
            quad = generate_quadrilateral()
            quad_poly_1 = Polygon(quad)
        quads.append(quad)
    return quads

def main():
    quads_1 = np.array(generate_convex_quads(200))
    quads_2 = np.array(generate_convex_quads(300))
    t1 = time.time()
    quads_1_torch = torch.tensor(quads_1).cuda()
    quads_2_torch = torch.tensor(quads_2).cuda()
    iou = quad_iou.calculateIoU(quads_1_torch, quads_2_torch)
    t2 = time.time()
    print("TIME TAKEN FOR CALCULATING IOU MATRIX WITH TORCH ", t2 - t1)
    iou_matrix = np.zeros((len(quads_1), len(quads_2)))
    t1 = time.time()
    for i, quad_1 in enumerate(quads_1):
        polygon_i = Polygon(quad_1)
        for j, quad_2 in enumerate(quads_2):
            polygon_j = Polygon(quad_2)
            iou = calc_iou_poly(polygon_i, polygon_j)
            iou_matrix[i][j] = iou
    t2 = time.time()
    print("TIME TAKEN FOR CALCULATING IOU MATRIX WITH SHAPELY ", t2 - t1)
    
if __name__ == "__main__":
    main()