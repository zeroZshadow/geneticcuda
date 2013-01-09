#ifndef __DNATRIANGLE__
#define __DNATRIANGLE__

#define min(x, y) x < y ? x : y
#define max(x, y) x > y ? x : y

__device__ inline void initTriangle(Triangle& triangleData, Settings& settings)
{
	int2 origin;
	initPoint(origin, settings);

	for( unsigned int j=0; j < 3; ++j)
	{
		triangleData.point[j].x = min(max(0, origin.x + randomBetween(-3, 3)), settings.imageInfo.imageWidth);
		triangleData.point[j].y = min(max(0, origin.x + randomBetween(-3, 3)), settings.imageInfo.imageHeight);
	}
}

#endif