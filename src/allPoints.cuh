#define MAX_INTERSECTION_POINTS 8
#define MAX_INSIDE_POINTS 8
#define MAX_ALL_POINTS 16
#include <torch/extension.h>

template <typename scalar_t>
__device__ inline void fillArrayWithInfinity(scalar_t points[][2], int num_points){
    for (int i = 0; i < num_points; i++){
        points[i][0] = INFINITY;
        points[i][1] = INFINITY;
    }
}

namespace allPoints {
    template <typename scalar_t>
    __device__ inline void fillPointsWithInfinity(scalar_t intersection_points[MAX_INTERSECTION_POINTS][2], scalar_t inside_points[MAX_INSIDE_POINTS][2], scalar_t all_points[MAX_ALL_POINTS][2]){
        fillArrayWithInfinity(intersection_points, MAX_INTERSECTION_POINTS);
        fillArrayWithInfinity(inside_points, MAX_INSIDE_POINTS);
        fillArrayWithInfinity(all_points, MAX_ALL_POINTS);
    }

    template <typename scalar_t>
    __device__ inline void copyIntersectionInsidePoints(scalar_t intersectionPoints[MAX_INTERSECTION_POINTS][2],
                                                        scalar_t insidePoints[MAX_INSIDE_POINTS][2],
                                                        scalar_t allPoints[MAX_ALL_POINTS][2]) {
        int nextAllPointIndex = 0;

        // Copy valid intersection points to allPoints
        for (int i = 0; i < MAX_INTERSECTION_POINTS; i++) {
            if (!isinf(intersectionPoints[i][0])) {
                allPoints[nextAllPointIndex][0] = intersectionPoints[i][0];
                allPoints[nextAllPointIndex][1] = intersectionPoints[i][1];
                nextAllPointIndex++;
            }
        }

        // Copy valid inside points to allPoints
        for (int i = 0; i < MAX_INSIDE_POINTS; i++){
            if (!isinf(insidePoints[i][0])){
                allPoints[nextAllPointIndex][0] = insidePoints[i][0];
                allPoints[nextAllPointIndex][1] = insidePoints[i][1];
                nextAllPointIndex++;
            }
        }
    }
}