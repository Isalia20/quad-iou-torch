#include <torch/extension.h>
#include "utils.cuh"

template <typename scalar_t>
__device__ inline scalar_t findMaxQuadCoordinate(const at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> box, Coordinate coord){
    // Find the maximum x-coordinate or y-coordinate of the quadrilateral based on the coord value
    scalar_t max_value = box[0][static_cast<int>(coord)];
    for (int i = 1; i < 4; ++i) {
        if (box[i][static_cast<int>(coord)] > max_value) {
            max_value = box[i][static_cast<int>(coord)];
        }
    }
    return max_value;
}

template <typename scalar_t>
__device__ inline int isPointInsideQuadrilateral(const Point<scalar_t>& point_to_check, const at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> box) {
    scalar_t max_x = findMaxQuadCoordinate(box, Coordinate::X);
    scalar_t max_y = findMaxQuadCoordinate(box, Coordinate::Y);
    // If the point's x-coordinate is greater than the max x-coordinate, it's outside
    if (point_to_check.x > max_x) return -1;
    if (point_to_check.y > max_y) return -1;

    // For each edge of the quadrilateral
    for (int i = 0; i < 4; i++) {
        // Get the current edge's start and end points
        Point<scalar_t> start_point = {box[i][0], box[i][1]}; 
        Point<scalar_t> end_point = {box[(i + 1) % 4][0], box[(i + 1) % 4][1]}; // Wrap around to the first point after the last
        // Calculate the cross product to determine where the point is in relation to the edge
        scalar_t cross_product = (start_point.y - point_to_check.y) * (end_point.x - point_to_check.x) -
                                (start_point.x - point_to_check.x) * (end_point.y - point_to_check.y);
        if (cross_product > 0) {
            return -1; // Point is outside the quadrilateral
        }
        else if (cross_product == 0) {
            return 0; // Point is on the boundary of the quadrilateral
        }
    }
    return 1; // Point is inside the quadrilateral
}

namespace insidePoints{
    template <typename scalar_t>
    __device__ inline void findPointsInside(const at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> quad_0, 
                                            const at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> quad_1, 
                                            at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> inside_points, 
                                            int maxPoints) {
        int numInsidePoints = 0;
        for (int i = 0; i < 4; i++) {
            Point<scalar_t> quad_0_point = {quad_0[i][0], quad_0[i][1]};
            if (isPointInsideQuadrilateral(quad_0_point, quad_1) == 1) {
                if (numInsidePoints < maxPoints) {
                    inside_points[numInsidePoints][0] = quad_0[i][0];
                    inside_points[numInsidePoints][1] = quad_0[i][1];
                    numInsidePoints++;
                }
            }
            Point<scalar_t> quad_1_point = {quad_1[i][0], quad_1[i][1]};
            if (isPointInsideQuadrilateral(quad_1_point, quad_0) == 1) {
                if (numInsidePoints < maxPoints) {
                    inside_points[numInsidePoints][0] = quad_1[i][0];
                    inside_points[numInsidePoints][1] = quad_1[i][1];
                    numInsidePoints++;
                }
            }
        }
    }
}