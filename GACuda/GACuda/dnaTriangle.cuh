#ifndef __DNATRIANGLE__
#define __DNATRIANGLE__

__device__ inline void initTriangle(Triangle& triangleData, Settings& settings)
{
	int2 origin;
	initPoint(origin, settings);

	for( unsigned int j=0; j < 3; ++j)
	{
		int moveRadius = settings.mutationRanges.pointMinMoveRange;
		triangleData.point[j].x = clamp(origin.x + randomBetween(-moveRadius, moveRadius), 0, settings.imageInfo.imageWidth-1);
		triangleData.point[j].y = clamp(origin.y + randomBetween(-moveRadius, moveRadius), 0, settings.imageInfo.imageHeight-1);
	}
}

#endif