#define MAX_ALL_POINTS 16
#include <torch/extension.h>
#include "utils.cuh"

namespace polygonArea{
    template <typename scalar_t>
    __device__ inline scalar_t calcPolygonArea(const at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> polygon) {
        scalar_t area = 0;
        int n = polygon.size(0);
        int j = 0; // Index of the previous valid vertex

        // Initialize the previous valid vertex
        #pragma unroll
        for (int i = 0; i < n; ++i) {
            if (!isinf(polygon[i][0]) && !isinf(polygon[i][1])) {
                j = i;
                break;
            }
        }

        // Calculate the sum for the Shoelace formula
        #pragma unroll
        for (int i = j + 1; i < n; ++i) {
            if (isinf(polygon[i][0]) && isinf(polygon[i][1])) continue; // Skip invalid vertices
            area += (polygon[j][0] * polygon[i][1] - polygon[i][0] * polygon[j][1]);
            j = i; // Update the index of the previous valid vertex
        }

        // Close the polygon loop if the last vertex is valid
        if (!isinf(polygon[j][0]) && !isinf(polygon[j][1]) && (!isinf(polygon[0][0]) && !isinf(polygon[0][1]))) {
            area += (polygon[j][0] * polygon[0][1] - polygon[0][0] * polygon[j][1]);
        }

        return fabs(area) / 2.0;
    }

    template <typename scalar_t>
    __device__ inline scalar_t calcPolygonArea(scalar_t polygon[MAX_ALL_POINTS][2]) {
        scalar_t area = 0;
        int n = MAX_ALL_POINTS;
        int j = 0; // Index of the previous valid vertex

        // Initialize the previous valid vertex
        #pragma unroll
        for (int i = 0; i < n; ++i) {
            if (!isinf(polygon[i][0]) && !isinf(polygon[i][1])) {
                j = i;
                break;
            }
        }

        // Calculate the sum for the Shoelace formula
        #pragma unroll
        for (int i = j + 1; i < n; ++i) {
            if (isinf(polygon[i][0]) && isinf(polygon[i][1])) continue; // Skip invalid vertices
            area += (polygon[j][0] * polygon[i][1] - polygon[i][0] * polygon[j][1]);
            j = i; // Update the index of the previous valid vertex
        }

        // Close the polygon loop if the last vertex is valid
        if (!isinf(polygon[j][0]) && !isinf(polygon[j][1]) && (!isinf(polygon[0][0]) && !isinf(polygon[0][1]))) {
            area += (polygon[j][0] * polygon[0][1] - polygon[0][0] * polygon[j][1]);
        }

        return fabs(area) / 2.0;
    }
}