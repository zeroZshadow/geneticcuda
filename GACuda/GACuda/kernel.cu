//Pre include
#include "settings.h"
#include "dnaStructures.cuh"
#include "cudaTools.cuh"

//Include dna helper functions
#include "dnaPoint.cuh"
#include "dnaTriangle.cuh"
#include "dnaColor.cuh"

#include "rasterizer.cuh"

#include <stdio.h>

__device__ __constant__ Settings g_settings;
texture<uchar4, cudaTextureType2D, cudaReadModeElementType> textureTarget;

__global__ void initProcess( void* counts, void* triangles, void* colors )
{
	int strainId = (blockDim.x * blockIdx.x) + threadIdx.x;

	int* triangleCounts = (int*)counts;
	Triangle* triangleData = (Triangle*)triangles;
	Color* colorData = (Color*)colors;

	//Set triangle count to 0 for strain
	unsigned int minTriangles = g_settings.mutationRanges.strainMinTriangles;
	triangleCounts[strainId] = minTriangles;

	//Initialize first X triangles and colors
	for( unsigned int i=0; i< minTriangles; ++i)
	{
		unsigned int index = interleavedIndex(strainId, i, g_settings.mutationRanges.strainMaxTriangles);
		initTriangle(triangleData[index], g_settings);
		initColor(colorData[index], g_settings);
	}
}

__global__ void renderProcess(  void* counts, void* triangles, void* colors, void* draw )
{
	//Setup basics
	int strainId = (blockDim.x * blockIdx.x) + threadIdx.x;
	int* triangleCounts = (int*)counts;
	Triangle* triangleData = (Triangle*)triangles;
	Color* colorData = (Color*)colors;

	//Drawbuffer
	int imagesize = g_settings.imageInfo.imageWidth * g_settings.imageInfo.imageHeight;
	int drawOffset = strainId * imagesize;
	uchar4* drawBuffer = &((uchar4*)draw)[drawOffset];

	//Initialize rasterizer
	int rasterStart = 0;
	int rasterEnd = g_settings.imageInfo.imageHeight-1;
	int2* rasterLines = new int2[g_settings.imageInfo.imageHeight];

	//Loop over triangles
	unsigned int triangleCount = triangleCounts[strainId];
	for( unsigned int i=0; i<triangleCount; ++i)
	{
		clearRaster(rasterLines, rasterStart, rasterEnd, g_settings);
		unsigned int index = interleavedIndex(strainId, i, g_settings.mutationRanges.strainMaxTriangles);

		for (int j=0; j<3; ++j)
		{
			int2& point1 = triangleData[index].point[i];
			int2& point2 = triangleData[index].point[(i+1)%3];
			passLine(point1, point2, rasterLines, rasterStart, rasterEnd, g_settings);
		}

		//Set color
		uchar4 color = colorData[index].components;
		float alphascale = (float)color.w / 255.0f;

		float4 fcolor;
		fcolor.x = color.x * alphascale;
		fcolor.y = color.y * alphascale;
		fcolor.z = color.z * alphascale;
		fcolor.w = color.w * alphascale;

		//Render triangle
		renderRaster(rasterLines, rasterStart, rasterEnd, drawBuffer, fcolor, g_settings);
	}


	//Cleanup
	delete[] rasterLines;
}

__global__ void fitnessProcess(  void* draw )
{
	//Calculate fitness of strain
}

extern "C" void launch_cudaInitialize(dim3 grid, dim3 block, void* counts, void* triangles, void* colors)
{
	initProcess<<< grid, block >>>( counts, triangles, colors );
}

extern "C" void launch_cudaRender(dim3 grid, dim3 block, void* counts, void* triangles, void* colors, void* draw)
{
	//ENGAGE!
	renderProcess<<< grid, block >>>( counts, triangles, colors, draw );
}

extern "C" void launch_cudaProcess(dim3 grid, dim3 block, void* draw)
{
	fitnessProcess<<< grid, block >>>( draw );
}

extern "C" void uploadSettings(Settings& settings)
{
	cudaError e = cudaMemcpyToSymbol (g_settings, &settings, sizeof(Settings));
	if ( e != cudaSuccess)
	{
		printf(cudaGetErrorString (e));
	}
}