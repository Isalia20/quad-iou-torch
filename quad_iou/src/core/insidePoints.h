#define MAX_ALL_POINTS 16
#include <torch/extension.h>
#include "utils.h"
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define HOST_DEVICE __host__ __device__
#define PRAGMA_UNROLL _Pragma("unroll")
#else
#define HOST_DEVICE
#define PRAGMA_UNROLL
#endif

template <typename scalar_t>
HOST_DEVICE inline scalar_t findMaxQuadCoordinate(const scalar_t *box, Coordinate coord){
    // Find the maximum x-coordinate or y-coordinate of the quadrilateral based on the coord value
    scalar_t max_value = box[static_cast<int>(coord)];
    PRAGMA_UNROLL
    for (int i = 1; i < 4; ++i) {
        if (box[i * 2 + static_cast<int>(coord)] > max_value) {
            max_value = box[i * 2 + static_cast<int>(coord)];
        }
    }
    return max_value;
}

template <typename scalar_t>
HOST_DEVICE inline int isPointInsideQuadrilateral(const Point<scalar_t>& point_to_check, const scalar_t *box) {
    scalar_t max_x = findMaxQuadCoordinate(box, Coordinate::X);
    scalar_t max_y = findMaxQuadCoordinate(box, Coordinate::Y);
    // If the point's x-coordinate is greater than the max x-coordinate, it's outside
    if (point_to_check.x > max_x) return -1;
    if (point_to_check.y > max_y) return -1;

    PRAGMA_UNROLL
    // For each edge of the quadrilateral
    for (int i = 0; i < 4; i++) {
        // Get the current edge's start and end points
        Point<scalar_t> start_point = {box[i * 2], box[i * 2 + 1]}; 
        Point<scalar_t> end_point = {box[((i + 1) % 4) * 2], box[((i + 1) % 4) * 2 + 1]}; // Wrap around to the first point after the last
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
    HOST_DEVICE inline void findPointsInside(const scalar_t *quad_0, 
                                            const scalar_t *quad_1, 
                                            scalar_t inside_points[MAX_ALL_POINTS][2],
                                            int numIntersections) {
        int numInsidePoints = numIntersections;
        PRAGMA_UNROLL
        for (int i = 0; i < 4; i++) {
            Point<scalar_t> quad_0_point = {quad_0[i * 2], quad_0[i * 2 + 1]};
            if (isPointInsideQuadrilateral(quad_0_point, quad_1) == 1) {
                if (numInsidePoints < MAX_ALL_POINTS) {
                    inside_points[numInsidePoints][0] = quad_0[i * 2];
                    inside_points[numInsidePoints][1] = quad_0[i * 2 + 1];
                    numInsidePoints++;
                }
            }
            Point<scalar_t> quad_1_point = {quad_1[i * 2], quad_1[i * 2 + 1]};
            if (isPointInsideQuadrilateral(quad_1_point, quad_0) == 1) {
                if (numInsidePoints < MAX_ALL_POINTS) {
                    inside_points[numInsidePoints][0] = quad_1[i * 2];
                    inside_points[numInsidePoints][1] = quad_1[i * 2 + 1];
                    numInsidePoints++;
                }
            }
        }
    }
}
