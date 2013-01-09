#ifndef __DNAPOINT__
#define __DNAPOINT__

__device__ void initPoint(int2& pointData, Settings& settings)
{
	pointData.x = randomBetween(0, settings.imageInfo.imageWidth-1);
	pointData.y = randomBetween(0, settings.imageInfo.imageHeight-1);
}

#endif