//Pre include
#include <helper_cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "settings.h"
#include "dnaStructures.cuh"
#include "cudaTools.cuh"

//Include dna helper functions
#include "dnaPoint.cuh"
#include "dnaTriangle.cuh"
#include "dnaColor.cuh"

//Include renderer
#include "rasterizer.cuh"

//Temporary for printing text
#include <stdio.h>

#include "ErrorCheck.h"

//Global variables
__device__ __constant__ Settings	g_settings;
__device__ __constant__ int*		g_triangleCounts;
__device__ __constant__ Triangle*	g_triangleData;
__device__ __constant__ Color*		g_colorData;
__device__ __constant__ uchar4*		g_bestBuffer;
__device__ __constant__ uchar4*		g_drawBuffer;
__device__ __constant__ int2*		g_rasterLines;
__device__ __constant__ uint2*		g_fitnessData;

texture<uchar4, 2, cudaReadModeElementType> texTarget;
__device__ unsigned int g_Generation = 0;


__device__ inline unsigned int getIndex()
{
	//Island offset + strain offset
	return (blockDim.x * blockIdx.x) + threadIdx.x;
}

__device__ inline unsigned int getGenerationStrainId()
{
	//Timeline offset + Island offset + strain offset
	const int idx = (blockDim.x * gridDim.x * (g_Generation&1)) + (blockDim.x * blockIdx.x) + threadIdx.x;
	return idx;
}

__device__ inline unsigned int getFutureStrainId()
{
	//Timeline+1 offset + Island offset + strain offset
	const int idx = (blockDim.x * gridDim.x * ((g_Generation+1)&1)) + (blockDim.x * blockIdx.x) + threadIdx.x;
	return idx;
}

__global__ void initProcess()
{
	const int strainId = getGenerationStrainId();

	//Grab global pointers
	int* triangleCounts = g_triangleCounts;
	Triangle* triangleData = g_triangleData;
	Color* colorData = g_colorData;

	//Set triangle count to 0 for strain
	unsigned int minTriangles = g_settings.mutationRanges.strainMinTriangles;
	triangleCounts[strainId] = minTriangles;


	//Initialize first X triangles and colors
	for (unsigned int i = 0; i < minTriangles; ++i)
	{
		unsigned int index = interleavedIndex(strainId, i, g_settings.mutationRanges.strainMaxTriangles);

		initTriangle(triangleData[index], g_settings);
		initColor(colorData[index], g_settings);
	}
}

__global__ void renderProcess()
{
	//Setup basics
	const int strainId = getGenerationStrainId();
	const int idx = getIndex();

	//Grab global pointers
	int* triangleCounts = g_triangleCounts;
	Triangle* triangleData = g_triangleData;
	Color* colorData = g_colorData;
	uchar4* drawBuffer = g_drawBuffer;
	int2* rasterLines = g_rasterLines;

	//Offset rasterbuffer
	rasterLines = &rasterLines[idx * g_settings.imageInfo.imageHeight];

	//Offset drawbuffer
	int imagesize = g_settings.imageInfo.imageWidth * g_settings.imageInfo.imageHeight;
	int drawOffset = idx * imagesize;
	drawBuffer = &drawBuffer[drawOffset];

	//Initialize raster
	int rasterStart = 0;
	int rasterEnd = g_settings.imageInfo.imageHeight-1;

	//Loop over triangles
	int triangleCount = triangleCounts[strainId];
	for (int i = 0; i < triangleCount; ++i)
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

__global__ void fitnessProcess()
{
	//Calculate fitness of strain
	int strainId = getGenerationStrainId();
	int idx = getIndex();
	unsigned int fitness = 0;

	//Grab global pointers
	uint2* fitnessBuffer = g_fitnessData;
	uchar4* drawBuffer = g_drawBuffer;

	//Drawbuffer
	const int height = g_settings.imageInfo.imageHeight;
	const int width = g_settings.imageInfo.imageWidth;
	const int drawOffset = idx * width * height;
	drawBuffer = &drawBuffer[drawOffset];

	int x = 0;
	int index = 0;
	for (int y = 0; y < height; ++y)
	{
		for (x = 0; x < width; ++x)
		{
			uchar4 utarget = tex2D(texTarget, x, y);
			uchar4 ustrain = drawBuffer[index];
			
			int r = utarget.x - ustrain.x;
			int g = utarget.y - ustrain.y;
			int b = utarget.z - ustrain.z;

			fitness += (unsigned int)(r*r + g*g + b*b);
			++index;
		}
	}

	fitnessBuffer[idx] = make_uint2(fitness, idx);
}

__global__ void evolveProcess()
{
	int strainId = getGenerationStrainId();
	int strainFutureId = getFutureStrainId();

	//Make new strain here based on score
	//TEMP Clone current strain to next generation
	int triangleCount = g_triangleCounts[strainId];
	g_triangleCounts[strainFutureId] = triangleCount;
	for (unsigned int i = 0; i < triangleCount; ++i)
	{
		unsigned int index = interleavedIndex(strainId, i, g_settings.mutationRanges.strainMaxTriangles);
		unsigned int indexFuture = interleavedIndex(strainFutureId, i, g_settings.mutationRanges.strainMaxTriangles);

		cloneTriangle(g_triangleData[indexFuture], g_triangleData[index]);
		cloneColor(g_colorData[strainFutureId], g_colorData[index]);
	}
}

__global__ void initRNG()
{
	//No generations used, calculate idx directly
	int idx = getIndex();
	curand_init(1337, idx, 0, &g_randState[idx]);
}

extern "C" void	launch_cudaSetupRNG(dim3 grid, dim3 block)
{
	initRNG<<<grid, block>>>();
}

extern "C" void launch_cudaInitialize( dim3 grid, dim3 block )
{
	initProcess<<< grid, block >>>();
}

extern "C" void launch_cudaRender(dim3 grid, dim3 block)
{
	//ENGAGE!
	renderProcess<<< grid, block >>>();
}

struct unit2_sort_x
{
	__host__ __device__
	bool operator()(uint2 x, uint2 y)
	{
		return x.x < y.x;
	}
};

extern "C" unsigned int launch_cudaFitness(dim3 grid, dim3 block, cudaArray* targetArray, void* fitnessData )
{
	//Bind texture
	cudaBindTextureToArray(texTarget, targetArray);
	struct cudaChannelFormatDesc desc;
	cudaGetChannelDesc(&desc, targetArray);

	//Run fitness function
	fitnessProcess<<< grid, block >>>();

	//Sort fitness array
	int arraysize = grid.x*block.x;
	thrust::device_ptr<uint2> dev_ptr = thrust::device_pointer_cast((uint2*)fitnessData);
	thrust::sort(dev_ptr, dev_ptr+arraysize, unit2_sort_x());

	//Since the best Score as at the start of the block, copy first Uint2 to host and return it
	uint2 bestId;
	cudaMemcpy(&bestId, fitnessData, sizeof(uint2), cudaMemcpyDeviceToHost);

	return bestId.y;
}

__host__ void increaseGeneration()
{
	unsigned int gen;
	cudaMemcpyFromSymbol(&gen, g_Generation, sizeof(unsigned int), 0);
	gen++;
	cudaMemcpyToSymbol(g_Generation, &gen, sizeof(unsigned int), 0);
}

extern "C" void launch_cudaEvolve(dim3 grid, dim3 block )
{
	//Generate new strains
	evolveProcess<<< grid, block >>>();
	increaseGeneration();
}

extern "C" void uploadConstants(Settings& settings,
	curandState* randState,
	void* counts,
	void* triangles,
	void* colors,
	void* best,
	void* draw,
	void* raster,
	void* fitness
)
{
	CudaSafeCall(cudaMemcpyToSymbol (g_settings,		&settings,	sizeof(Settings)));
	CudaSafeCall(cudaMemcpyToSymbol (g_randState,		&randState, sizeof(curandState*)));
	CudaSafeCall(cudaMemcpyToSymbol (g_triangleCounts,	&counts,	sizeof(int*)));
	CudaSafeCall(cudaMemcpyToSymbol (g_triangleData,	&triangles, sizeof(Triangle*)));
	CudaSafeCall(cudaMemcpyToSymbol (g_colorData,		&colors,	sizeof(Color*)));
	CudaSafeCall(cudaMemcpyToSymbol (g_bestBuffer,		&best,		sizeof(uchar4*)));
	CudaSafeCall(cudaMemcpyToSymbol (g_drawBuffer,		&draw,		sizeof(uchar4*)));
	CudaSafeCall(cudaMemcpyToSymbol (g_rasterLines,		&raster,	sizeof(uint2*)));
	CudaSafeCall(cudaMemcpyToSymbol (g_fitnessData,		&fitness,	sizeof(uint2*)));
}