import random
import torch
import numpy as np
import time
from shapely.geometry import Polygon
from quad_iou import calculate_iou
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

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

def shapely_iou(polys1, polys2):
    intersection_areas = [[intersected_poly.area for intersected_poly in poly] for poly in [i.intersection(polys2) for i in polys1]]
    poly1_areas = [poly.area for poly in polys1]
    poly2_areas = [poly.area for poly in polys2]
    ious = [
        intersection_areas[i][j] / (area_1 + area_2 - intersection_areas[i][j])
        for i, area_1 in enumerate(poly1_areas)
        for j, area_2 in enumerate(poly2_areas)
    ]
    return ious


def benchmark_iou_calculations(num_pairs):
    quad1_list = [generate_random_quadrilateral() for _ in range(num_pairs)]
    quad2_list = [generate_random_quadrilateral() for _ in range(num_pairs)]
    quad1_tensors = torch.tensor(quad1_list).reshape(-1, 4, 2).double()
    quad2_tensors = torch.tensor(quad2_list).reshape(-1, 4, 2).double()
    polys1 = [Polygon(quad) for quad in quad1_list]
    polys2 = [Polygon(quad) for quad in quad2_list]
    
    start_time = time.time()
    iou_matrix = calculate_iou(quad1_tensors, quad2_tensors, True)
    quad_iou_time = time.time() - start_time
    
    t1 = time.time()
    ious = shapely_iou(polys1, polys2)
    t2 = time.time()
    return quad_iou_time, t2 - t1

def main():
    num_pairs_list = [i for i in range(1, 1002, 100)]
    results = {"num_pairs": [], "quad_iou_time": [], "shapely_iou_time": []}

    for num_pairs in num_pairs_list:
        quad_iou_time, shapely_iou_time = benchmark_iou_calculations(num_pairs)
        results["num_pairs"].append(num_pairs ** 2)
        results["quad_iou_time"].append(quad_iou_time)
        results["shapely_iou_time"].append(shapely_iou_time)
        print(f"{num_pairs} pairs -> quad_iou: {quad_iou_time:.6f} s, shapely: {shapely_iou_time:.6f} s")

    plt.figure()
    plt.plot(results["num_pairs"], results["quad_iou_time"], label='Quad IOU Time')
    plt.plot(results["num_pairs"], results["shapely_iou_time"], label='Shapely IOU Time')
    plt.xlabel('Number of Quadrilateral pairs for iou calculation')
    plt.ylabel('Time (s)')
    plt.title('IOU Calculation Benchmark')
    plt.legend()
    plt.grid(True)
    formatter = FuncFormatter(lambda x, _: f'{int(x):,}')
    plt.gca().xaxis.set_major_formatter(formatter)

    plt.savefig("iou_benchmark_plot.png")

if __name__ == "__main__":
    main()
