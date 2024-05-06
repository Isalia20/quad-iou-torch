#include <torch/extension.h>

template <typename scalar_t>
__device__ inline scalar_t* findMinMaxQuadCoordinate(const at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> box, Coordinate coord){
    // Find the maximum x-coordinate or y-coordinate of the quadrilateral based on the coord value
    // First value will be minimum and second will be maximum, hence min_max
    static __device__ scalar_t min_max_values[2];

    min_max_values[0] = box[0][static_cast<int>(coord)];
    min_max_values[1] = box[0][static_cast<int>(coord)];

    for (int i = 1; i < 4; ++i) {
        // Min value
        if (box[i][static_cast<int>(coord)] < min_max_values[0]) {
            min_max_values[0] = box[i][static_cast<int>(coord)];
        }
        // Max value
        if (box[i][static_cast<int>(coord)] > min_max_values[1]) {
            min_max_values[1] = box[i][static_cast<int>(coord)];
        }
    }
    return min_max_values;
}

template <typename scalar_t>
__device__ inline bool checkSimpleIntersection(at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> quad_0,
                                               at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int> quad_1){
    // Function to check a very simple intersection. If one quad's x's and y's are not overlapping with another's x and y's 
    // we know that intersection area will be 0 and we avoid other calculations in kernel
    // quad_0
    scalar_t* min_max_x_0 = findMinMaxQuadCoordinate(quad_0, Coordinate::X);
    scalar_t* min_max_y_0 = findMinMaxQuadCoordinate(quad_0, Coordinate::Y);
    // quad_1
    scalar_t* min_max_x_1 = findMinMaxQuadCoordinate(quad_1, Coordinate::X);
    scalar_t* min_max_y_1 = findMinMaxQuadCoordinate(quad_1, Coordinate::Y);
    // Check for overlap in the x and y dimensions
    bool overlap_x = (min_max_x_0[0] <= min_max_x_1[1]) && (min_max_x_0[1] >= min_max_x_1[0]);
    bool overlap_y = (min_max_y_0[0] <= min_max_y_1[1]) && (min_max_y_0[1] >= min_max_y_1[0]);
    return overlap_x && overlap_y;
}