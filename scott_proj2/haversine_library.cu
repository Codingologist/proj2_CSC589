#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

__global__ void haversine_distance_kernel(int size, const double *x1, const double *y1,
                                          const double *x2, const double *y2, double *dist)
{
    // use any references to compute haversine distance bewtween (x1,y1) and (x2,y2), given in vectors/arrays
    // e.g., https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula
    int ind = blockIdx.x * blockDim.x + threadIdx.x; // block index
    int R = 6371;

    if (ind < size)
    {
        double dLat = (y2[ind] - y1[ind]) * (M_PI / 180);
        double dLon = (x2[ind] - x1[ind]) * (M_PI / 180);
        double y1Rad = y1[ind] * (M_PI / 180);
        double y2Rad = y2[ind] * (M_PI / 180);

        double a = sin(dLat / 2) * sin(dLat / 2) + cos(y1Rad) * cos(y2Rad) * sin(dLon / 2) * sin(dLon / 2);
        double c = 2 * atan2(sqrt(a), sqrt(1 - a));
        dist[ind] = R * c;
    }
}

void run_kernel(int size, const double *x1, const double *y1, const double *x2, const double *y2, double *dist)

{
    dim3 dimBlock(1024);
    printf("in run_kernel dimBlock.x=%d\n", dimBlock.x);

    dim3 dimGrid(ceil((double)size / dimBlock.x));

    haversine_distance_kernel<<<dimGrid, dimBlock>>>(size, x1, y1, x2, y2, dist);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::stringstream strstr;
        strstr << "run_kernel launch failed" << std::endl;
        strstr << "dimBlock: " << dimBlock.x << ", " << dimBlock.y << std::endl;
        strstr << "dimGrid: " << dimGrid.x << ", " << dimGrid.y << std::endl;
        strstr << cudaGetErrorString(error);
        throw strstr.str();
    }
}