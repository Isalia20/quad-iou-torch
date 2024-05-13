#include <torch/extension.h>

template <typename scalar_t>
__device__ inline void findMinMaxQuadCoordinate(const at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int>& box, 
                                                Coordinate coord, scalar_t& min_val, scalar_t& max_val) {
    // Initialize min and max with the first point's coordinate
    int coord_index = static_cast<int>(coord);
    min_val = box[0][coord_index];
    max_val = box[0][coord_index];

    // Loop through the remaining points
    for (int i = 1; i < 4; ++i) {
        scalar_t val = box[i][coord_index];
        if (val < min_val) {
            min_val = val;
        }
        if (val > max_val) {
            max_val = val;
        }
    }
}

namespace simpleIntersectCheck {
    template <typename scalar_t>
    __device__ inline bool checkSimpleIntersection(const at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int>& quad_0,
                                                   const at::TensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int>& quad_1) {
        // Function to check a very simple intersection. If one quad's x's and y's are not overlapping with another's x and y's 
        // we know that intersection area will be 0 and we avoid other calculations in kernel
        scalar_t min_x_0, max_x_0, min_y_0, max_y_0;
        scalar_t min_x_1, max_x_1, min_y_1, max_y_1;

        // Find min/max for both quads
        findMinMaxQuadCoordinate(quad_0, Coordinate::X, min_x_0, max_x_0);
        findMinMaxQuadCoordinate(quad_0, Coordinate::Y, min_y_0, max_y_0);
        findMinMaxQuadCoordinate(quad_1, Coordinate::X, min_x_1, max_x_1);
        findMinMaxQuadCoordinate(quad_1, Coordinate::Y, min_y_1, max_y_1);

        // Check for overlap in the x and y dimensions
        bool overlap_x = (min_x_0 <= max_x_1) && (max_x_0 >= min_x_1);
        bool overlap_y = (min_y_0 <= max_y_1) && (max_y_0 >= min_y_1);

        return overlap_x && overlap_y;
    }
}