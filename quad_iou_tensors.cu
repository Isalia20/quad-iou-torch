#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
struct Point {
    scalar_t x, y;
};

enum class Coordinate {
    X,
    Y
};

template <typename scalar_t>
__device__ void findPointsInside(const at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> quad_0, 
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

template <typename scalar_t>
__device__ void findIntersectionPoints(const at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> quad_0, 
                                      const at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> quad_1, 
                                      at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> intersections, 
                                      int maxIntersections) {
    int numIntersections = 0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            Point<scalar_t> intersection;
            Point<scalar_t> quad_0_point_one = {quad_0[i][0], quad_0[i][1]};
            Point<scalar_t> quad_0_point_two = {quad_0[(i + 1) % 4][0], quad_0[(i + 1) % 4][1]};
            Point<scalar_t> quad_1_point_one = {quad_1[j][0], quad_1[j][1]};
            Point<scalar_t> quad_1_point_two = {quad_1[(j + 1) % 4][0], quad_1[(j + 1) % 4][1]};
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
                if (!alreadyExists && numIntersections < maxIntersections) {
                    intersections[numIntersections][0] = intersection.x;
                    intersections[numIntersections][1] = intersection.y;
                    numIntersections++;
                }
            }
        }
    }
}

template <typename scalar_t>
__device__ int orientation(const Point<scalar_t>& p, const Point<scalar_t>& q, const Point<scalar_t>& r) {
    scalar_t val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    if (fabsf(val) < 1e-10) return 0;  // colinear
    return (val > 0) ? 1 : 2;  // clockwise
}

template <typename scalar_t>
__device__ scalar_t findMaxQuadCoordinate(const at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> box, Coordinate coord){
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
__device__ scalar_t findMinQuadCoordinate(const at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> box, Coordinate coord){
    // Find the minimum x-coordinate or y-coordinate of the quadrilateral based on the coord value
    scalar_t min_value = box[0][static_cast<int>(coord)];
    for (int i = 1; i < 4; ++i) {
        if (box[i][static_cast<int>(coord)] < min_value) {
            min_value = box[i][static_cast<int>(coord)];
        }
    }
    return min_value;
}

template <typename scalar_t>
__device__ int isPointInsideQuadrilateral(const Point<scalar_t>& point_to_check, const at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> box) {
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

template <typename scalar_t>
__device__ bool doIntersect(const Point<scalar_t>& p1, const Point<scalar_t>& q1, const Point<scalar_t>& p2, const Point<scalar_t>& q2, Point<scalar_t>& intersection) {
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

        if (abs(determinant) < 1e-10) {
            return false; // The lines are parallel. This is simplified
                          // by returning false here.
        } else {
            intersection.x = (b2 * c1 - b1 * c2) / determinant;
            intersection.y = (a1 * c2 - a2 * c1) / determinant;
            return true;
        }
    }

    // Special Cases
    // p1, q1 and p2 are colinear and p2 lies on segment p1q1
    if (o1 == 0 && onSegment(p1, p2, q1)) {
        intersection = p2;
        return true;
    }

    // p1, q1 and q2 are colinear and q2 lies on segment p1q1
    if (o2 == 0 && onSegment(p1, q2, q1)) {
        intersection = q2;
        return true;
    }

    // p2, q2 and p1 are colinear and p1 lies on segment p2q2
    if (o3 == 0 && onSegment(p2, p1, q2)) {
        intersection = p1;
        return true;
    }

    // p2, q2 and q1 are colinear and q1 lies on segment p2q2
    if (o4 == 0 && onSegment(p2, q1, q2)) {
        intersection = q1;
        return true;
    }

    return false; // Doesn't fall in any of the above cases
}

template <typename scalar_t>
__device__ bool onSegment(const Point<scalar_t>& p, const Point<scalar_t>& q, const Point<scalar_t>& r) {
    return q.x <= max(p.x, r.x) && q.x >= min(p.x, r.x) &&
           q.y <= max(p.y, r.y) && q.y >= min(p.y, r.y);
}

template <typename scalar_t>
__device__ bool arePointsEqual(const Point<scalar_t>& p1, const Point<scalar_t>& p2) {
    return fabsf(p1.x - p2.x) < 1e-10 && fabsf(p1.y - p2.y) < 1e-10;
}

template <typename scalar_t>
__device__ Point<scalar_t> findCentroid(const at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> points) {
    Point<scalar_t> centroid = {0.0, 0.0};
    int valid_point_counter = 0;
    for (int i = 0; i < points.size(0); i++) {
        if (points[i][0] != -1.0 && points[i][1] != -1.0){
            centroid.x += points[i][0];
            centroid.y += points[i][1];
            valid_point_counter++;
        }
    }
    centroid.x /= valid_point_counter;
    centroid.y /= valid_point_counter;
    return centroid;
}

template <typename scalar_t>
__device__ Point<scalar_t> findCentroid(torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> points) {
    Point<scalar_t> centroid = {0.0, 0.0};
    int valid_point_counter = 0;
    for (int i = 0; i < points.size(0); i++) {
        if (points[i][0] != -1.0 && points[i][1] != -1.0){
            centroid.x += points[i][0];
            centroid.y += points[i][1];
            valid_point_counter++;
        }
    }
    centroid.x /= valid_point_counter;
    centroid.y /= valid_point_counter;
    return centroid;
}

template <typename scalar_t>
__device__ scalar_t polygonArea(const at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> polygon) {
    scalar_t area = 0;
    int n = polygon.size(0);
    int j = 0; // Index of the previous valid vertex

    // Initialize the previous valid vertex
    for (int i = 0; i < n; ++i) {
        if (polygon[i][0] != -1 && polygon[i][1] != -1) {
            j = i;
            break;
        }
    }

    // Calculate the sum for the Shoelace formula
    for (int i = j + 1; i < n; ++i) {
        if (polygon[i][0] == -1 && polygon[i][1] == -1) continue; // Skip invalid vertices
        area += (polygon[j][0] * polygon[i][1] - polygon[i][0] * polygon[j][1]);
        j = i; // Update the index of the previous valid vertex
    }

    // Close the polygon loop if the last vertex is valid
    if (polygon[j][0] != -1 && polygon[j][1] != -1 && (polygon[0][0] != -1 && polygon[0][1] != -1)) {
        area += (polygon[j][0] * polygon[0][1] - polygon[0][0] * polygon[j][1]);
    }

    return fabs(area) / 2.0;
}

template <typename scalar_t>
__device__ void copyIntersectionInsidePoints(at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> intersectionPoints,
                                             at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> insidePoints,
                                             at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> allPoints){
    int nextInsidePointIndex = 0;

    for (int i = 0; i < intersectionPoints.size(0); i++){
        if (intersectionPoints[i][0] != -1 && intersectionPoints[i][1] != -1){
            allPoints[i][0] = intersectionPoints[i][0];
            allPoints[i][1] = intersectionPoints[i][1];
        }
    }

    for (int i = 0; i < allPoints.size(0); i++){
        if (allPoints[i][0] == -1 && allPoints[i][1] == -1){ // if points haven't been changed
            // Find the next unused inside point
            while (nextInsidePointIndex < insidePoints.size(0) && 
                   (insidePoints[nextInsidePointIndex][0] == -1 || insidePoints[nextInsidePointIndex][1] == -1)) {
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
__device__ scalar_t computeAngle(const Point<scalar_t>& centroid, const Point<scalar_t>& p) {
    // Use atan2f for float and atan2 for double
    return (sizeof(scalar_t) == sizeof(double)) ? atan2(p.y - centroid.y, p.x - centroid.x) : atan2f(p.y - centroid.y, p.x - centroid.x);
}

template <typename scalar_t>
__device__ bool comparePoints(const Point<scalar_t>& p1, const Point<scalar_t>& p2, const Point<scalar_t>& centroid) {
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

template<typename scalar_t>
__device__ void swapPoints(at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> points, int i){
    scalar_t tempX = points[i][0];
    scalar_t tempY = points[i][1];
    points[i][0] = points[i + 1][0];
    points[i][1] = points[i + 1][1];
    points[i + 1][0] = tempX;
    points[i + 1][1] = tempY;
}

// Sorts a vector of points in clockwise order(can be upgraded to a better sorting algorithm)
template <typename scalar_t>
__device__ void sortPointsClockwise(at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> points) {
    // Calculate the centroid of the points
    Point<scalar_t> centroid = findCentroid(points);
    
    bool swapped = true; // Initialize swapped to true to enter the loop
    int n = points.size(0);
    while (swapped) {
        swapped = false; // Set swapped to false at the beginning of the loop
        for (int i = 0; i < n - 1; i++) {
            // Skip points where both x and y are -1
            if (points[i][0] == -1 && points[i][1] == -1) continue;
            if (points[i + 1][0] == -1 && points[i + 1][1] == -1) continue;
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

template <typename scalar_t>
__device__ bool checkSimpleIntersection(at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> quad_0,
                                        at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> quad_1){
    // Function to check a very simple intersection. If one quad's x's and y's are not overlapping with another's x and y's 
    // we know that intersection area will be 0 and we avoid other calculations in kernel
    // quad_0
    scalar_t max_x_0 = findMaxQuadCoordinate(quad_0, Coordinate::X);
    scalar_t max_y_0 = findMaxQuadCoordinate(quad_0, Coordinate::Y);
    scalar_t min_x_0 = findMinQuadCoordinate(quad_0, Coordinate::X);
    scalar_t min_y_0 = findMinQuadCoordinate(quad_0, Coordinate::Y);
    // quad_1
    scalar_t max_x_1 = findMaxQuadCoordinate(quad_1, Coordinate::X);
    scalar_t max_y_1 = findMaxQuadCoordinate(quad_1, Coordinate::Y);
    scalar_t min_x_1 = findMinQuadCoordinate(quad_1, Coordinate::X);
    scalar_t min_y_1 = findMinQuadCoordinate(quad_1, Coordinate::Y);

    // Check for overlap in the x and y dimensions
    bool overlap_x = (min_x_0 <= max_x_1) && (max_x_0 >= min_x_1);
    bool overlap_y = (min_y_0 <= max_y_1) && (max_y_0 >= min_y_1);
    return overlap_x && overlap_y;
}

template <typename scalar_t>
__device__ scalar_t intersectionAreaCuda(at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> quad_0,
                                         at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> quad_1,
                                         at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> intersectionPoints,
                                         at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> insidePoints,
                                         at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> allPoints,
                                         const int MAX_INTERSECTIONS
                                         ){
    sortPointsClockwise(quad_0);
    sortPointsClockwise(quad_1);

    // If we know that quad_0 and quad_1 are not intersecting even a tiny bit(minimum enclosing box check) 
    // we can skip below calculation altogether
    if (!checkSimpleIntersection(quad_0, quad_1)) return 0.0;

    findIntersectionPoints(quad_0, quad_1, intersectionPoints, MAX_INTERSECTIONS);
    findPointsInside(quad_0, quad_1, insidePoints, MAX_INTERSECTIONS);
    copyIntersectionInsidePoints(intersectionPoints, insidePoints, allPoints);
    sortPointsClockwise(allPoints);
    scalar_t intersectArea = polygonArea(allPoints);
    return intersectArea;
}

template <typename scalar_t>
__device__ scalar_t unionAreaCuda(const at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> quad_0,\
                                  const at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> quad_1, 
                                  scalar_t intersectArea){
    return polygonArea(quad_0) + polygonArea(quad_1) - intersectArea;
}

template <typename scalar_t>
__device__ scalar_t calculateIoUCuda(const at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> quad_0,
                                     const at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> quad_1,
                                     at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> intersectionPoints,
                                     at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> insidePoints,
                                     at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> allPoints,
                                     const int MAX_INTERSECTIONS){
    const scalar_t epsilon = 0.00001;
    scalar_t intersect_area = intersectionAreaCuda(quad_0, quad_1, intersectionPoints, insidePoints, allPoints, MAX_INTERSECTIONS);
    return intersect_area / (epsilon + unionAreaCuda(quad_0, quad_1, intersect_area));
}

template <typename scalar_t>
__global__ void calculateIoUCudaKernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> quad_0,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> quad_1,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> iou_matrix,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> intersectionPoints,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> insidePoints,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> allPoints,
    const int MAX_INTERSECTIONS
    ) {

    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = blockIdx.y * blockDim.y + threadIdx.y;

    if ((idx1 < quad_0.size(0)) && (idx2 < quad_1.size(0))){
        iou_matrix[idx1][idx2] = calculateIoUCuda(quad_0[idx1], 
                                                  quad_1[idx2], 
                                                  intersectionPoints[idx1][idx2],
                                                  insidePoints[idx1][idx2],
                                                  allPoints[idx1][idx2],
                                                  MAX_INTERSECTIONS
                                                  );
    }
}


torch::Tensor calculateIoUCudaTorch(torch::Tensor quad_0, torch::Tensor quad_1) {
    TORCH_CHECK(quad_0.numel() > 0 && quad_1.numel() > 0, "Input tensors must not empty");
    
    const int MAX_INTERSECTIONS = 8; // 8 intersections max
    // Create an output tensor and tensors for calculating intersection area
    torch::Tensor iou_matrix = torch::zeros({quad_0.size(0), quad_1.size(0)}, quad_0.options());
    torch::Tensor intersectionPoints = torch::ones({quad_0.size(0), quad_1.size(0), MAX_INTERSECTIONS, 2}, quad_0.options()) * -1;
    torch::Tensor insidePoints = torch::ones({quad_0.size(0), quad_1.size(0), MAX_INTERSECTIONS, 2}, quad_0.options()) * -1;
    torch::Tensor allPoints = torch::ones({quad_0.size(0), quad_1.size(0), MAX_INTERSECTIONS * 2, 2}, quad_0.options()) * -1;

    // Calculate the number of blocks and threads
    dim3 blockSize(16, 16);
    dim3 gridSize((quad_0.size(0) + blockSize.x - 1) / blockSize.x, (quad_1.size(0) + blockSize.y - 1) / blockSize.y);

    AT_DISPATCH_FLOATING_TYPES(quad_0.scalar_type(), "calculateIoUCudaTorch", ([&] {
        calculateIoUCudaKernel<scalar_t><<<gridSize, blockSize>>>(
            quad_0.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            quad_1.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            iou_matrix.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            intersectionPoints.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            insidePoints.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            allPoints.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            MAX_INTERSECTIONS
            );
        }));

    // Synchronize to wait for the computation to finish
    cudaDeviceSynchronize();
    // Check for any errors launching the kernel
    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return torch::tensor({}); // Return an empty tensor if there was an error
    }

    return iou_matrix;
}
