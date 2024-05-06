#define MAX_INTERSECTION_POINTS 8
#define MAX_INSIDE_POINTS 8
#define MAX_ALL_POINTS 16
#include <torch/extension.h>
#include <cmath>
#include "polygonArea.cuh"
#include "insidePoints.cuh"
#include "intersectionPoints.cuh"
#include "sortPoints.cuh"
#include "allPoints.cuh"
#include "simpleIntersectCheck.cuh"
#include "checks.cuh"


template <typename scalar_t>
__device__ inline scalar_t intersectionArea(at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> quad_0,
                                            at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> quad_1
                                            ){
    // If we know that quad_0 and quad_1 are not intersecting even a tiny bit(minimum enclosing box check) 
    // we can skip below calculation altogether
    if (!simpleIntersectCheck::checkSimpleIntersection(quad_0, quad_1)) return 0.0;

    scalar_t intersection_points[MAX_INTERSECTION_POINTS][2];
    scalar_t inside_points[MAX_INSIDE_POINTS][2];
    scalar_t all_points[MAX_ALL_POINTS][2];
    allPoints::fillPointsWithInfinity(intersection_points, inside_points, all_points);

    intersectionPoints::findIntersectionPoints(quad_0, quad_1, intersection_points);
    insidePoints::findPointsInside(quad_0, quad_1, inside_points);
    allPoints::copyIntersectionInsidePoints(intersection_points, inside_points, all_points);
    sortPoints::sortPointsClockwise(all_points);
    scalar_t intersectArea = polygonArea::calcPolygonArea(all_points);
    return intersectArea;
}

template <typename scalar_t>
__device__ inline scalar_t unionArea(const at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> quad_0,
                                         const at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> quad_1, 
                                         int quad_0_idx,
                                         int quad_1_idx,
                                         int quad_0_size,
                                         scalar_t* polygonAreas,
                                         scalar_t intersectArea){
    return polygonAreas[quad_0_idx] + polygonAreas[quad_0_size + quad_1_idx] - intersectArea;
}

template <typename scalar_t>
__device__ inline scalar_t calculateIoU(const at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> quad_0,
                                        const at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> quad_1,
                                        int quad_0_idx,
                                        int quad_1_idx,
                                        int quad_0_size,
                                        scalar_t* polygonAreas){
    const scalar_t epsilon = 0.00001;

    scalar_t intersect_area = intersectionArea(quad_0, quad_1);
    return intersect_area / (unionArea(quad_0, quad_1, quad_0_idx, quad_1_idx, quad_0_size, polygonAreas, intersect_area) + epsilon);
}

template <typename scalar_t>
__global__ void calculateIoUKernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> quad_0,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> quad_1,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> iou_matrix,
    scalar_t* polygonAreas
    ) {

    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = blockIdx.y * blockDim.y + threadIdx.y;

    if ((idx1 < quad_0.size(0)) && (idx2 < quad_1.size(0))){
        scalar_t iou = calculateIoU(quad_0[idx1], 
                                    quad_1[idx2], 
                                    idx1,
                                    idx2,
                                    quad_0.size(0),
                                    polygonAreas
                                    );
        // avoid access if iou is 0, since iou matrix is initialized to 0s
        if (iou != 0.0){
            iou_matrix[idx1][idx2] = iou;
        }
    }
}

template <typename scalar_t>
__global__ void sortPointsKernel(
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> quad_0,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> quad_1
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < quad_0.size(0)){
        sortPoints::sortPointsClockwise(quad_0[idx]);
    } else if (idx < (quad_0.size(0) + quad_1.size(0))){
        sortPoints::sortPointsClockwise(quad_1[idx - quad_0.size(0)]);
    }
}

template <typename scalar_t>
__global__ void polygonAreaCalculationKernel(
    scalar_t* polygonAreas,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> quad_0,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> quad_1
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < quad_0.size(0)){
        polygonAreas[idx] = polygonArea::calcPolygonArea(quad_0[idx]);
    } else if (idx < (quad_0.size(0) + quad_1.size(0))){
        polygonAreas[idx] = polygonArea::calcPolygonArea(quad_1[idx - quad_0.size(0)]);
    }
}

torch::Tensor calculateIoUCudaTorch(torch::Tensor quad_0, torch::Tensor quad_1) {
    checks::check_tensor_validity(quad_0, quad_1);
    
    // Create an output tensor
    torch::Tensor iou_matrix = torch::zeros({quad_0.size(0), quad_1.size(0)}, quad_0.options());

    AT_DISPATCH_FLOATING_TYPES(quad_0.scalar_type(), "calculateIoUCudaTorch", ([&] {
        scalar_t* polygonAreas_d;
        cudaMalloc((void**)&polygonAreas_d, (quad_0.size(0) + quad_1.size(0)) * sizeof(scalar_t));

        dim3 blockSize(16, 16);
        dim3 gridSize((quad_0.size(0) + blockSize.x - 1) / blockSize.x, (quad_1.size(0) + blockSize.y - 1) / blockSize.y);
        dim3 blockSizeQuad(128, 1, 1);
        dim3 gridSizeQuad((quad_0.size(0) + quad_1.size(0) + blockSizeQuad.x - 1) / blockSizeQuad.x, 1, 1);

        sortPointsKernel<scalar_t><<<gridSizeQuad, blockSizeQuad>>>(
            quad_0.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            quad_1.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>()
        );

        polygonAreaCalculationKernel<scalar_t><<<gridSizeQuad, blockSizeQuad>>>(
            polygonAreas_d,
            quad_0.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            quad_1.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>()
        );

        calculateIoUKernel<scalar_t><<<gridSize, blockSize>>>(
            quad_0.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            quad_1.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            iou_matrix.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            polygonAreas_d
        );
        cudaFree(polygonAreas_d);
    }));
    return iou_matrix;
}
