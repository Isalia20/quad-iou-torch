#define MAX_INTERSECTION_POINTS 8
#define MAX_ALL_POINTS 16
#define EPSILON 1e-6
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
HOST_DEVICE inline bool arePointsEqual(const Point<scalar_t>& p1, const Point<scalar_t>& p2) {
    return fabsf(p1.x - p2.x) < EPSILON && fabsf(p1.y - p2.y) < EPSILON;
}

template <typename scalar_t>
HOST_DEVICE inline int orientation(const Point<scalar_t>& p, const Point<scalar_t>& q, const Point<scalar_t>& r) {
    scalar_t val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    if (fabsf(val) < EPSILON) return 0;  // colinear
    return (val > 0) ? 1 : 2;  // clockwise
}

template <typename scalar_t>
HOST_DEVICE inline bool doIntersect(const Point<scalar_t>& p1, const Point<scalar_t>& q1, const Point<scalar_t>& p2, const Point<scalar_t>& q2, Point<scalar_t>& intersection) {
    // Find the four orientations needed for general and
    // special cases
    int o1 = orientation(p1, q1, p2);
    int o2 = orientation(p1, q1, q2);
    int o3 = orientation(p2, q2, p1);
    int o4 = orientation(p2, q2, q1);

    // General case
    if (o1 != o2 && o3 != o4) {
        // Line AB represented as a1x + b1y = c1
        scalar_t a1 = q1.y - p1.y;
        scalar_t b1 = p1.x - q1.x;
        scalar_t c1 = a1 * (p1.x) + b1 * (p1.y);

        // Line CD represented as a2x + b2y = c2
        scalar_t a2 = q2.y - p2.y;
        scalar_t b2 = p2.x - q2.x;
        scalar_t c2 = a2 * (p2.x) + b2 * (p2.y);

        scalar_t determinant = a1 * b2 - a2 * b1;
        if (fabsf(determinant) < EPSILON) {
            return false; // The lines are parallel
        } else {
            intersection.x = (b2 * c1 - b1 * c2) / determinant;
            intersection.y = (a1 * c2 - a2 * c1) / determinant;
            return true;
        }
    }
    
    return false; // Doesn't fall in any of the above cases
}

namespace intersectionPoints{    
    template <typename scalar_t>
    HOST_DEVICE inline int findIntersectionPoints(const scalar_t *quad_0, 
                                                  const scalar_t *quad_1, 
                                                  scalar_t intersections[MAX_ALL_POINTS][2]) {
        int numIntersections = 0;
        PRAGMA_UNROLL
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                Point<scalar_t> intersection;
                Point<scalar_t> quad_0_point_one = {quad_0[i * 2], quad_0[i * 2 + 1]};
                Point<scalar_t> quad_0_point_two = {quad_0[((i + 1) % 4) * 2], quad_0[((i + 1) % 4) * 2 + 1]};
                Point<scalar_t> quad_1_point_one = {quad_1[j * 2], quad_1[j * 2 + 1]};
                Point<scalar_t> quad_1_point_two = {quad_1[((j + 1) % 4) * 2], quad_1[((j + 1) % 4) * 2 + 1]};
                if (doIntersect(quad_0_point_one, quad_0_point_two, quad_1_point_one, quad_1_point_two, intersection)) {
                    // Check if this intersection is already in the intersections array
                    bool alreadyExists = false;
                    for (int k = 0; k < numIntersections; ++k) {
                        Point<scalar_t> intersection_point = {intersections[k][0], intersections[k][1]};
                        if (arePointsEqual(intersection_point, intersection)) {
                            alreadyExists = true;
                            break;
                        }
                    }
                    if (!alreadyExists && numIntersections < MAX_INTERSECTION_POINTS) {
                        intersections[numIntersections][0] = intersection.x;
                        intersections[numIntersections][1] = intersection.y;
                        numIntersections++;
                    }
                }
            }
        }
        return numIntersections;
    }
}
