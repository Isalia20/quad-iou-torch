#include <torch/extension.h>

namespace allPoints {
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
}