#ifndef __CUDATOOLS__
#define __CUDATOOLS__

#include <curand.h>
#include <curand_kernel.h>

__device__ __constant__ curandState* g_randState;

__device__ float clamp(float x, float a, float b)
{
    return min(max(x, a), b);
}

__device__ inline int wrap(int kX, int const kLowerBound, int const kUpperBound)
{
    int range_size = kUpperBound - kLowerBound + 1;

    if (kX < kLowerBound)
        kX += range_size * ((kLowerBound - kX) / range_size + 1);

    return kLowerBound + (kX - kLowerBound) % range_size;
}

__device__ inline unsigned int fastrand()
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	return curand(&g_randState[idx]);
}

__device__ inline int randomBetween(int aMin, int aMax)
{
	if (aMin == aMax)
		return aMin;

	int result = wrap(fastrand(), aMin, aMax);

	return result;
}

__device__ inline bool willMutate(int rate)
{
	return (randomBetween(0, rate) == 0);
}

__device__ inline unsigned int interleavedIndex(unsigned int strainId, unsigned int triangleId, unsigned int maxUnits)
{
	return maxUnits * triangleId + strainId;
}

#endif