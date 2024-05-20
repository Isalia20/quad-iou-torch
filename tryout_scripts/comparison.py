import random
import torch
import numpy as np
import time
from shapely.geometry import Polygon
from quad_iou import calculate_iou

def generate_random_point():
    return (random.uniform(0, 100), random.uniform(0, 100))

def generate_random_quadrilateral():
    while True:
        quadrilateral = [generate_random_point() for _ in range(4)]
        if is_convex(quadrilateral):
            return quadrilateral

def shapely_iou(poly1, poly2):
    intersection_area = poly1.intersection(poly2).area
    return intersection_area / (poly1.area + poly2.area - intersection_area)

def is_convex(quadrilateral):
    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    signs = []
    for i in range(4):
        o, a, b = quadrilateral[i], quadrilateral[(i + 1) % 4], quadrilateral[(i + 2) % 4]
        signs.append(np.sign(cross_product(o, a, b)))

    return all(s == signs[0] for s in signs)

def main():
    quad1 = generate_random_quadrilateral()
    quad2 = generate_random_quadrilateral()
    quad1_tensor = torch.tensor(quad1).reshape(-1, 4, 2).double()
    quad2_tensor = torch.tensor(quad2).reshape(-1, 4, 2).double()
    poly1 = Polygon(quad1)
    poly2 = Polygon(quad2)

    for _ in range(10):
        calculate_iou(quad1_tensor, quad2_tensor, True)
        shapely_iou(poly1, poly2)

    num_iterations = 1000
    start_time = time.time()
    for _ in range(num_iterations):
        iou_quad_iou = calculate_iou(quad1_tensor, quad2_tensor, False)
    end_time = time.time()
    avg_time_quad = (end_time - start_time) / num_iterations
    print(f"quad_iou average time: {avg_time_quad:.10f} seconds")

    start_time = time.time()
    for _ in range(num_iterations):
        iou_shapely = shapely_iou(poly1, poly2)
    end_time = time.time()
    avg_time_shapely = (end_time - start_time) / num_iterations
    print(f"shapely average time: {avg_time_shapely:.10f} seconds")
    print(f"quad_iou IoU: {iou_quad_iou[0]}")
    print(f"shapely IoU: {iou_shapely}")

if __name__ == "__main__":
    main()
