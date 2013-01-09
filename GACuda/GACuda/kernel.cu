//Pre include
#include <helper_cuda.h>
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

__global__ void renderProcess(  void* counts, void* triangles, void* colors, void* draw, void* raster )
{
	//Setup basics
	int strainId = (blockDim.x * blockIdx.x) + threadIdx.x;
	int* triangleCounts = (int*)counts;
	Triangle* triangleData = (Triangle*)triangles;
	Color* colorData = (Color*)colors;
	uchar4* drawBuffer = (uchar4*)draw;

	//raster buffer
	int2* rasterLines = (int2*)raster;
	rasterLines = &rasterLines[strainId * g_settings.imageInfo.imageHeight];

	//Drawbuffer
	int imagesize = g_settings.imageInfo.imageWidth * g_settings.imageInfo.imageHeight;
	int drawOffset = strainId * imagesize;
	drawBuffer = &drawBuffer[drawOffset];

	//Initialize rasterizer
	int rasterStart = 0;
	int rasterEnd = g_settings.imageInfo.imageHeight-1;

	//Loop over triangles
	int triangleCount = triangleCounts[strainId];
	for( int i=0; i<triangleCount; ++i)
	{
		clearRaster(rasterLines, rasterStart, rasterEnd, g_settings);
		unsigned int index = interleavedIndex(strainId, i, g_settings.mutationRanges.strainMaxTriangles);

		
		for (int j=0; j<3; ++j)
		{
			int2& point1 = triangleData[index].point[j];
			int2& point2 = triangleData[index].point[(j+1)%3];
			passLine(point1, point2, rasterLines, rasterStart, rasterEnd, g_settings);
		}
		//Set color
		uchar4 color = colorData[index].components;
		float alphascale = (float)color.w / 255.0f;

		float4 fcolor;
		fcolor.x = color.x * alphascale;
		fcolor.y = color.y * alphascale;
		fcolor.z = color.z * alphascale;
		fcolor.w = 0;

		//Render triangle
		renderRaster(rasterLines, rasterStart, rasterEnd, drawBuffer, fcolor, g_settings, strainId);
	}
}

__global__ void fitnessProcess(  void* draw )
{
	//Calculate fitness of strain
}

__global__ void initRNG()
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(1337, idx, 0, &g_randState[idx]);
}

extern "C" void	launch_cudaSetupRNG(dim3 grid, dim3 block)
{
	initRNG<<<grid, block>>>();
}

extern "C" void launch_cudaInitialize(dim3 grid, dim3 block, void* counts, void* triangles, void* colors)
{
	initProcess<<< grid, block >>>( counts, triangles, colors );
}

extern "C" void launch_cudaRender(dim3 grid, dim3 block, void* counts, void* triangles, void* colors, void* draw, void* raster)
{
	//ENGAGE!
	renderProcess<<< grid, block >>>( counts, triangles, colors, draw, raster );
}

extern "C" void launch_cudaFitness(dim3 grid, dim3 block, void* draw, void* best)
{
	fitnessProcess<<< grid, block >>>( draw );
}

extern "C" void uploadConstants(Settings& settings, curandState* randState)
{
	cudaError e = cudaMemcpyToSymbol (g_settings, &settings, sizeof(Settings));
	if ( e != cudaSuccess)
	{
		printf(cudaGetErrorString (e));
	}

	e = cudaMemcpyToSymbol (g_randState, &randState, sizeof(curandState*));
	if ( e != cudaSuccess)
	{
		printf(cudaGetErrorString (e));
	}
}