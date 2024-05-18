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
    quads_1 = np.array(generate_convex_quads(2000))
    quads_2 = np.array(generate_convex_quads(200))
    quads_1_torch = torch.tensor(quads_1).cuda()
    quads_2_torch = torch.tensor(quads_2).cuda()
    t1 = time.time()
    iou = quad_iou.calculate_iou(quads_1_torch, quads_2_torch, True)
    t2 = time.time()
    time_quad_iou = t2 - t1
    iou_matrix = np.zeros((len(quads_1), len(quads_2)))
    t1 = time.time()
    quads_1_poly = [Polygon(i) for i in quads_1]
    quads_2_poly = [Polygon(i) for i in quads_2]
    for i, quad_1_poly in enumerate(quads_1_poly):
        for j, quad_2_poly in enumerate(quads_2_poly):
            iou = calc_iou_poly(quad_1_poly, quad_2_poly)
            iou_matrix[i][j] = iou
    t2 = time.time()
    time_shapely = t2 - t1
    return time_quad_iou, time_shapely
    
if __name__ == "__main__":
    times_0 = []
    times_1 = []
    for i in range(10):
        time_quad_iou, time_shapely = main()
        times_0.append(time_quad_iou)
        times_1.append(time_shapely)
    print(f"CUDA IOU TOOK ON AVERAGE {sum(times_0) / len(times_0)} seconds")
    print(f"SHAPELY IOU TOOK ON AVERAGE {sum(times_1) / len(times_1)} seconds")
