#ifndef __CUDATOOLS__
#define __CUDATOOLS__

#include <curand.h>
#include <curand_kernel.h>

__device__ __constant__ curandState* g_randState;

__device__ float clamp(float x, float a, float b)
{
    return min(max(x, a), b);
}

__device__ inline int wrap(int kX, int const kLowerBound, const int kUpperBound)
{
	const int range_size = kUpperBound - kLowerBound + 1;

    if (kX < kLowerBound)
        kX += range_size * ((kLowerBound - kX) / range_size + 1);

    return kLowerBound + (kX - kLowerBound) % range_size;
}

__device__ inline unsigned int fastrand()
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	return curand(&g_randState[idx]);
}

__device__ inline int randomBetween(const int aMin, const int aMax)
{
	if (aMin == aMax)
		return aMin;

	return wrap(fastrand(), aMin, aMax);
}

__device__ inline bool willMutate(const int rate)
{
	return ((fastrand() % rate) == 0);
}

__device__ inline unsigned int interleavedIndex(const unsigned int strainId, const unsigned int triangleId, const unsigned int maxUnits)
{
	return maxUnits * triangleId + strainId;
}

#endif