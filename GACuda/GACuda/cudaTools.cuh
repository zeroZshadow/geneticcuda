#ifndef __CUDATOOLS__
#define __CUDATOOLS__

__device__ unsigned int g_seed = 0;

__device__ inline void fastrand_srand(unsigned int seed)
{
	g_seed = seed;
}

__device__ inline int fastrand()
{
	g_seed = (214013 * g_seed + 2531011);
	return (g_seed >> 16) & 0x7FFF;
}

__device__ inline int randomBetween(int min, int max)
{
	if (min == max)
		return min;

	return (fastrand()%(max-min))+min;
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