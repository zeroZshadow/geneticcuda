#ifndef __RASTERIZER__
#define __RASTERIZER__

#include <stdio.h>

#define min(x, y) x < y ? x : y
#define max(x, y) x > y ? x : y

__device__ int clamp(int x, int a, int b)
{
    return max(a, min(b, x));
}

__device__ void clearRaster(int2* rasterLines, int& rasterStart, int& rasterEnd, Settings& settings)
{
	for(int line = rasterStart; line <= rasterEnd; ++line){
		rasterLines[line].x = settings.imageInfo.imageWidth+1;
		rasterLines[line].y = -1;
	}
	rasterStart = settings.imageInfo.imageHeight;
	rasterEnd = 0;
}

__device__ void passLine(int2& p1, int2& p2, int2* rasterLines, int& rasterStart, int& rasterEnd, Settings& settings)
{
	//Process a line into the raster
	int2 point1 = p1;
	int2 point2 = p2;
	if (point1.y > point2.y){ //Make sure L1 is the highest (lowest actually) point
		point1 = p2;
		point2 = p1;
	}

	
	//dy
	float dy = (float)point2.y - (float)point1.y;
	if (dy == 0.0f) dy = 1.0f;
	float rcpDY = 1.0f / dy;

	//Get Slopes
	float	dx = (point2.x - point1.x) * rcpDY;

	int iY1 = point1.y;
	int iY2 = point2.y;

	//Init and subpixel correction
	const float fix = 1.0f - ((float)point1.y - iY1);
	float x = (float)point1.x + dx * fix;
	++iY1;

	for (int y = iY1; y <= iY2; y++){
		rasterStart = min(y, rasterStart);
		rasterEnd = max(y, rasterEnd);

		rasterLines[y].x = min(x, rasterLines[y].x);
		rasterLines[y].y = max(x, rasterLines[y].y);

		x += dx;
	}
}

__device__ void renderRaster(int2* rasterLines, int& rasterStart, int& rasterEnd, uchar4* drawBuffer, float4& color, Settings& settings)
{
	int width = settings.imageInfo.imageWidth;

	//Draw the current raster
	uchar4* buffer = drawBuffer;
	buffer += width * rasterStart;

	for(int y=rasterStart; y <= rasterEnd; ++y){
		const int iXmin = rasterLines[y].x;
		const int iXmax = rasterLines[y].y;

		const int length = iXmax - iXmin;
		for(int i=0; i <= length; ++i){
			uchar4 dst = *(buffer + iXmin + i);
			uchar4 result;
			result.x = clamp(dst.x + color.x, 0, 255);
			result.x = clamp(dst.y + color.y, 0, 255);
			result.x = clamp(dst.z + color.z, 0, 255);
			result.x = clamp(dst.w + color.w, 0, 255);

			*(buffer + iXmin + i) = result;
		}
		buffer += width;
	}
}

#endif