#define MAX_ALL_POINTS 16
#include <torch/extension.h>
#include "utils.h"
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

namespace polygonArea{
    template <typename scalar_t>
    HOST_DEVICE inline scalar_t calcQuadrilateralArea(const scalar_t *quadrilateral) {
        scalar_t area = 0.0;
        const int vertices = 4;

        // Calculate the sum for the Gaussian formula
        #pragma unroll
        for (int i = 1, j = 0; i < vertices; ++i) {
            area += quadrilateral[j * 2] * quadrilateral[i * 2 + 1] - 
                    quadrilateral[i * 2] * quadrilateral[j * 2 + 1];
            j = i; // Update the index of the previous vertex
        }
        
        // Complete the loop by adding the last term
        area += quadrilateral[6] * quadrilateral[1] - 
                quadrilateral[0] * quadrilateral[7];

        return fabs(area) / 2.0;
    }

    template <typename scalar_t>
    HOST_DEVICE inline scalar_t calcPolygonArea(scalar_t polygon[MAX_ALL_POINTS][2]) {
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

        // Calculate the sum for the Gaussian formula
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
