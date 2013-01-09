#ifndef __RASTERIZER__
#define __RASTERIZER__

#include <stdio.h>

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

	//Get Slopes
	const float	dx = (point2.x - point1.x) /dy;

	const int iY1 = point1.y;
	const int iY2 = point2.y;
	float x = (float)point1.x;

	rasterStart = min(iY1, rasterStart);
	rasterEnd = max(iY2, rasterEnd);

	for (int y = iY1; y <= iY2; ++y){
		rasterLines[y].x = min((int)x, rasterLines[y].x);
		rasterLines[y].y = max((int)x, rasterLines[y].y);
		x += dx;
	}
}

__device__ void renderRaster(int2* rasterLines, int& rasterStart, int& rasterEnd, uchar4* drawBuffer, float4& color, Settings& settings, int id)
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
			float4 fdst = make_float4(dst.x, dst.y, dst.z, dst.w);
			uchar4 result = make_uchar4(
				clamp(fdst.x + color.x, 0.0f, 255.0f),
				clamp(fdst.y + color.y, 0.0f, 255.0f),
				clamp(fdst.z + color.z, 0.0f, 255.0f),
				255
			);

			*(buffer + iXmin + i) = result;		
		}
		buffer += width;
	}
}

#endif