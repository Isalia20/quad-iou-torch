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
    __device__ inline void copyIntersectionInsidePoints(at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> intersectionPoints,
                                                        at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> insidePoints,
                                                        at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> allPoints){
        int nextInsidePointIndex = 0;

        for (int i = 0; i < intersectionPoints.size(0); i++){
            if (!isinf(intersectionPoints[i][0]) && !isinf(intersectionPoints[i][1])){
                allPoints[i][0] = intersectionPoints[i][0];
                allPoints[i][1] = intersectionPoints[i][1];
            }
        }

        for (int i = 0; i < allPoints.size(0); i++){
            if (isinf(allPoints[i][0]) && isinf(allPoints[i][1])){ // if points haven't been changed
                // Find the next unused inside point
                while (nextInsidePointIndex < insidePoints.size(0) && 
                    (isinf(insidePoints[nextInsidePointIndex][0]) || isinf(insidePoints[nextInsidePointIndex][1]))) {
                    nextInsidePointIndex++;
                }
                // If there's an unused inside point, use it to fill allPoints
                if (nextInsidePointIndex < insidePoints.size(0)) {
                    allPoints[i][0] = insidePoints[nextInsidePointIndex][0];
                    allPoints[i][1] = insidePoints[nextInsidePointIndex][1];
                    nextInsidePointIndex++; // Move to the next inside point for the next iteration
                } else {
                    // No more unused inside points available
                    break;
                }
            }
        }
    }

    template <typename scalar_t>
    __device__ inline void copyIntersectionInsidePoints(scalar_t intersectionPoints[MAX_INTERSECTION_POINTS][2],
                                                        scalar_t insidePoints[MAX_INSIDE_POINTS][2],
                                                        scalar_t allPoints[MAX_ALL_POINTS][2]){
        int nextInsidePointIndex = 0;

        for (int i = 0; i < MAX_INTERSECTION_POINTS; i++){
            if (!isinf(intersectionPoints[i][0]) && !isinf(intersectionPoints[i][1])){
                allPoints[i][0] = intersectionPoints[i][0];
                allPoints[i][1] = intersectionPoints[i][1];
            }
        }

        for (int i = 0; i < MAX_ALL_POINTS; i++){
            if (isinf(allPoints[i][0]) && isinf(allPoints[i][1])){ // if points haven't been changed
                // Find the next unused inside point
                while (nextInsidePointIndex < MAX_INSIDE_POINTS && 
                    (isinf(insidePoints[nextInsidePointIndex][0]) || isinf(insidePoints[nextInsidePointIndex][1]))) {
                    nextInsidePointIndex++;
                }
                // If there's an unused inside point, use it to fill allPoints
                if (nextInsidePointIndex < MAX_INSIDE_POINTS) {
                    allPoints[i][0] = insidePoints[nextInsidePointIndex][0];
                    allPoints[i][1] = insidePoints[nextInsidePointIndex][1];
                    nextInsidePointIndex++; // Move to the next inside point for the next iteration
                } else {
                    // No more unused inside points available
                    break;
                }
            }
        }
    }
}