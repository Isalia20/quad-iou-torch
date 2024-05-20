#define MAX_INTERSECTION_POINTS 8
#define MAX_INSIDE_POINTS 8
#define MAX_ALL_POINTS 16
#include <torch/extension.h>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif


template <typename scalar_t>
HOST_DEVICE inline void fillArrayWithInfinity(scalar_t points[][2], int num_points){
    #pragma unroll
    for (int i = 0; i < num_points; i++){
        points[i][0] = INFINITY;
        points[i][1] = INFINITY;
    }
}

namespace allPoints {
    template <typename scalar_t>
    HOST_DEVICE inline void fillPointsWithInfinity(scalar_t all_points[MAX_ALL_POINTS][2]){
        fillArrayWithInfinity(all_points, MAX_ALL_POINTS);
    }
}
