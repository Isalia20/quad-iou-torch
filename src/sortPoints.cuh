#define MAX_ALL_POINTS 16
#include <torch/extension.h>
#include "utils.cuh"


template <typename scalar_t>
__device__ inline scalar_t computeAngle(const Point<scalar_t>& centroid, const Point<scalar_t>& p) {
    // Use atan2f for float and atan2 for double
    return (sizeof(scalar_t) == sizeof(double)) ? atan2(p.y - centroid.y, p.x - centroid.x) : atan2f(p.y - centroid.y, p.x - centroid.x);
}

template <typename scalar_t>
__device__ inline Point<scalar_t> findCentroid(scalar_t *points) {
    Point<scalar_t> centroid = {0.0, 0.0};
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        centroid.x += points[i * 2];
        centroid.y += points[i * 2 + 1];
    }
    centroid.x /= 4.0;
    centroid.y /= 4.0;
    return centroid;
}

template <typename scalar_t>
__device__ inline Point<scalar_t> findCentroid(scalar_t points[MAX_ALL_POINTS][2]) {
    Point<scalar_t> centroid = {0.0, 0.0};
    int valid_point_counter = 0;
    #pragma unroll
    for (int i = 0; i < MAX_ALL_POINTS; i++) {
        if (!isinf(points[i][0]) && !isinf(points[i][1])){
            centroid.x += points[i][0];
            centroid.y += points[i][1];
            valid_point_counter++;
        }
    }
    centroid.x /= valid_point_counter;
    centroid.y /= valid_point_counter;
    return centroid;
}

template<typename scalar_t>
__device__ inline void swapPoints(scalar_t *points, int i){
    scalar_t tempX = points[i * 2];
    scalar_t tempY = points[i * 2 + 1];
    points[i * 2] = points[(i + 1) * 2];
    points[i * 2 + 1] = points[(i + 1) * 2 + 1];
    points[(i + 1) * 2] = tempX;
    points[(i + 1) * 2 + 1] = tempY;
}

template<typename scalar_t>
__device__ inline void swapPoints(scalar_t points[MAX_ALL_POINTS][2], int i){
    scalar_t tempX = points[i][0];
    scalar_t tempY = points[i][1];
    points[i][0] = points[i + 1][0];
    points[i][1] = points[i + 1][1];
    points[i + 1][0] = tempX;
    points[i + 1][1] = tempY;
}

template <typename scalar_t>
__device__ inline bool comparePoints(const Point<scalar_t>& p1, const Point<scalar_t>& p2, const Point<scalar_t>& centroid) {
    const scalar_t EPSILON = 1e-6;

    scalar_t angle1 = computeAngle(centroid, p1);
    scalar_t angle2 = computeAngle(centroid, p2);

    if (fabs(angle1 - angle2) < EPSILON) {
        scalar_t dist1 = (p1.x - centroid.x) * (p1.x - centroid.x) +
                         (p1.y - centroid.y) * (p1.y - centroid.y);
        scalar_t dist2 = (p2.x - centroid.x) * (p2.x - centroid.x) +
                         (p2.y - centroid.y) * (p2.y - centroid.y);
        return dist1 < dist2;
    }
    return angle1 < angle2;
}

namespace sortPoints{
    template <typename scalar_t>
    __device__ inline void sortQuadPointsClockwise(scalar_t *points) {
        // Calculate the centroid of the points
        Point<scalar_t> centroid = findCentroid(points);
        
        bool swapped = true; // Initialize swapped to true to enter the loop
        int n = 4;
        while (swapped) {
            swapped = false; // Set swapped to false at the beginning of the loop
            for (int i = 0; i < n - 1; i++) {
                Point<scalar_t> p1 = {points[i * 2], points[i * 2 + 1]};
                Point<scalar_t> p2 = {points[(i + 1) * 2], points[(i + 1) * 2 + 1]};

                // Using the comparison function to determine if the points are out of order
                if (!comparePoints(p1, p2, centroid)) {
                    // Swap points if they are out of order
                    swapPoints(points, i);
                    swapped = true; // Indicate a swap occurred
                }
            }
            // Decrement n because the last element is now guaranteed to be in place
            --n;
        }
    }

    template <typename scalar_t>
    __device__ inline void sortPointsClockwise(scalar_t points[MAX_ALL_POINTS][2]) {
        // Calculate the centroid of the points
        Point<scalar_t> centroid = findCentroid(points);
        
        bool swapped = true; // Initialize swapped to true to enter the loop
        int n = MAX_ALL_POINTS;
        while (swapped) {
            swapped = false; // Set swapped to false at the beginning of the loop
            for (int i = 0; i < n - 1; i++) {
                // Skip points where both x and y are inf
                if (isinf(points[i][0]) && isinf(points[i][1])) continue;
                if (isinf(points[i + 1][0]) && isinf(points[i + 1][1])) continue;
                Point<scalar_t> p1 = {points[i][0], points[i][1]};
                Point<scalar_t> p2 = {points[i + 1][0], points[i + 1][1]};

                // Using the comparison function to determine if the points are out of order
                if (!comparePoints(p1, p2, centroid)) {
                    // Swap points if they are out of order
                    swapPoints(points, i);
                    swapped = true; // Indicate a swap occurred
                }
            }
            // Decrement n because the last element is now guaranteed to be in place
            --n;
        }
    }
}