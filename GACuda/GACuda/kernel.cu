#define IRASTER
#define IDRAW

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

__device__ inline unsigned int indexToGenerationIndex(const unsigned int index)
{
	return (blockDim.x * gridDim.x * ((g_Generation)&1)) + index;
}

__device__ inline unsigned int indexToFutureIndex(const unsigned int index)
{
	return (blockDim.x * gridDim.x * ((g_Generation+1)&1)) + index;
}

__device__ inline unsigned int getGenerationStrainId()
{
	//Timeline offset + Island offset + strain offset
	return (blockDim.x * gridDim.x * (g_Generation&1)) + (blockDim.x * blockIdx.x) + threadIdx.x;
}

__device__ inline unsigned int getFutureStrainId()
{
	//Timeline+1 offset + Island offset + strain offset
	return (blockDim.x * gridDim.x * ((g_Generation+1)&1)) + (blockDim.x * blockIdx.x) + threadIdx.x;
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

	//Offset rasterbuffer
#ifndef IRASTER
	int2* rasterLines = &g_rasterLines[idx * g_settings.imageInfo.imageHeight];
#else
	int2* rasterLines = &g_rasterLines[idx];
#endif

	//Offset drawbuffer
	const int imagesize = g_settings.imageInfo.imageWidth * g_settings.imageInfo.imageHeight;
#ifndef IDRAW
	const int drawOffset = idx * imagesize;
	uchar4* drawBuffer = &g_drawBuffer[drawOffset];
#else
	uchar4* drawBuffer = &g_drawBuffer[idx];
#endif

	//Initialize raster
	int rasterStart = 0;
	int rasterEnd = g_settings.imageInfo.imageHeight-1;

	//Loop over triangles
	const int triangleCount = triangleCounts[strainId];
	for (int i = 0; i < triangleCount; ++i)
	{
		clearRaster(rasterLines, rasterStart, rasterEnd, g_settings);
		const unsigned int index = interleavedIndex(strainId, i, g_settings.mutationRanges.strainMaxTriangles);

		
		for (int j=0; j<3; ++j)
		{
			int2& point1 = triangleData[index].point[j];
			int2& point2 = triangleData[index].point[(j+1)%3];
			passLine(point1, point2, rasterLines, rasterStart, rasterEnd, g_settings);
		}
		//Set color
		const uchar4 color = colorData[index].components;
		const float alphascale = (float)color.w / 255.0f;

		float4 fcolor = make_float4(
			color.x * alphascale,
			color.y * alphascale,
			color.z * alphascale,
			0
		);

		//Render triangle
		renderRaster(rasterLines, rasterStart, rasterEnd, drawBuffer, fcolor, g_settings, strainId);
	}
}

__global__ void fitnessProcess()
{
	//Calculate fitness of strain
	const int strainId = getGenerationStrainId();
	const int idx = getIndex();
	const int strains = blockDim.x * gridDim.x;
	unsigned int fitness = 0;

	//Grab global pointers
	uint2* fitnessBuffer = g_fitnessData;

	//Drawbuffer
	const int height = g_settings.imageInfo.imageHeight;
	const int width = g_settings.imageInfo.imageWidth;

#ifndef IDRAW
	const int drawOffset = idx * width * height;
	const uchar4* drawBuffer = &g_drawBuffer[drawOffset];
#else
	const uchar4* drawBuffer = &g_drawBuffer[idx];
#endif

	int x = 0;
	int y = (height / blockDim.y) * threadIdx.y;
	const int ymax = (height / blockDim.y) * (threadIdx.y+1);
#ifndef IDRAW
	int index = y * width;
#else
	int index = strains * y * width;
#endif
	for (; y < ymax; ++y)
	{
		for (x = 0; x < width; ++x)
		{
			const uchar4 utarget = tex2D(texTarget, x, y);
			const uchar4 ustrain = drawBuffer[index];
			
			const int r = utarget.x - ustrain.x;
			const int g = utarget.y - ustrain.y;
			const int b = utarget.z - ustrain.z;

			fitness += (unsigned int)(r*r + g*g + b*b);
#ifndef IDRAW
			++index;
#else
			index += strains;
#endif
		}
	}
	
	fitnessBuffer[idx] = make_uint2(0, idx);
	__syncthreads();
	atomicAdd((unsigned int*)&fitnessBuffer[idx], fitness);
}

__global__ void evolveProcess()
{
	const int strainId = getGenerationStrainId();
	const int strainFutureId = getFutureStrainId();
	const int idx = getIndex();

	const int maxstrains = g_settings.generationInfo.strainCount * g_settings.generationInfo.islandCount;

	//Tournament selection
	//Pick a random strain from this island
	//(strainId + randomBetween(0, maxstrains-1)) % maxstrains;
	const int randomNumber = (idx + randomBetween(0, maxstrains-1)) % maxstrains;//(threadIdx.x + randomBetween(0, g_settings.generationInfo.strainCount-1)) % g_settings.generationInfo.strainCount;
	const int randomIdx = randomNumber;//(blockDim.x * blockIdx.x) + randomNumber; //scale id to be global
	const int randomGenerationId = indexToGenerationIndex(randomIdx);

	//Compare scores
	const int winnerId = (g_fitnessData[idx].x < g_fitnessData[randomIdx].x) ? strainId : randomGenerationId;
	const bool mutate = (winnerId != strainId);

	//Clone winning strain to future strain
	const int triangleCount = g_triangleCounts[winnerId];
	g_triangleCounts[strainFutureId] = triangleCount;
	for (int i = 0; i < triangleCount; ++i)
	{
		const unsigned int index = interleavedIndex(winnerId, i, g_settings.mutationRanges.strainMaxTriangles);
		const unsigned int indexFuture = interleavedIndex(strainFutureId, i, g_settings.mutationRanges.strainMaxTriangles);

		cloneTriangle(g_triangleData[indexFuture],	g_triangleData[index]);
		cloneColor(g_colorData[indexFuture],		g_colorData[index]);

		if(mutate)
		{
			//MUTATE POINTS
			for( int pointIdx = 0; pointIdx < 3; ++pointIdx)
			{
				int2& point = g_triangleData[indexFuture].point[pointIdx];
				if ( willMutate(g_settings.mutationRates.pointMinMoveMutationRate))
				{
					point.x = clamp(point.x + randomBetween(-g_settings.mutationRanges.pointMinMoveRange, g_settings.mutationRanges.pointMinMoveRange), 0, g_settings.imageInfo.imageWidth-1);
					point.y = clamp(point.y + randomBetween(-g_settings.mutationRanges.pointMinMoveRange, g_settings.mutationRanges.pointMinMoveRange), 0, g_settings.imageInfo.imageHeight-1);

				}

				if ( willMutate(g_settings.mutationRates.pointMidMoveMutationRate))
				{
					point.x = clamp(point.x + randomBetween(-g_settings.mutationRanges.pointMidMoveRange, g_settings.mutationRanges.pointMinMoveRange), 0, g_settings.imageInfo.imageWidth-1);
					point.y = clamp(point.y + randomBetween(-g_settings.mutationRanges.pointMidMoveRange, g_settings.mutationRanges.pointMinMoveRange), 0, g_settings.imageInfo.imageHeight-1);
				}

				if ( willMutate(g_settings.mutationRates.pointMaxMoveMutationRate))
				{
					point.x = fastrand() % g_settings.imageInfo.imageWidth;
					point.y = fastrand() % g_settings.imageInfo.imageHeight;
				}
			}

			//MUTATE COLORS
			uchar4& components = g_colorData[indexFuture].components;
			if ( willMutate(g_settings.mutationRates.redMutationRate))
			{
				components.x = randomBetween(g_settings.mutationRanges.redRangeMin, g_settings.mutationRanges.redRangeMax);
			}
			if ( willMutate(g_settings.mutationRates.greenMutationRate))
			{
				components.y = randomBetween(g_settings.mutationRanges.greenRangeMin, g_settings.mutationRanges.greenRangeMax);
			}
			if ( willMutate(g_settings.mutationRates.blueMutationRate))
			{
				components.z = randomBetween(g_settings.mutationRanges.blueRangeMin, g_settings.mutationRanges.blueRangeMax);
			}
			if ( willMutate(g_settings.mutationRates.alphaMutationRate))
			{
				components.w = randomBetween(g_settings.mutationRanges.alphaRangeMin, g_settings.mutationRanges.alphaRangeMax);
			}
		}
	}

	//Mutate if this strain was a loser
	if(mutate)
	{
		if (willMutate(g_settings.mutationRates.strainAddTriangleMutationRate))
		{
			int count = g_triangleCounts[strainFutureId];
			if (count < g_settings.mutationRanges.strainMaxTriangles)
			{
				const unsigned int indexFutureAdd = interleavedIndex(strainFutureId, count, g_settings.mutationRanges.strainMaxTriangles);

				initTriangle(g_triangleData[indexFutureAdd], g_settings);
				initColor(g_colorData[indexFutureAdd], g_settings);

				g_triangleCounts[strainFutureId]++;
			}
		}
		if ( willMutate(g_settings.mutationRates.strainRemoveTriangleMutationRate))
		{
			int count = g_triangleCounts[strainFutureId];
			if (count > g_settings.mutationRanges.strainMinTriangles)
			{
				//Choose random triangle to remove
				const unsigned int triangleIdx = fastrand() % count;

				//To remove a triangle, simply clone the last triangle in the list OVER the removed triangle
				const unsigned int indexFutureEnd = interleavedIndex(strainFutureId, count-1, g_settings.mutationRanges.strainMaxTriangles);
				const unsigned int indexFutureRemove = interleavedIndex(strainFutureId, triangleIdx, g_settings.mutationRanges.strainMaxTriangles);

				cloneTriangle(g_triangleData[indexFutureRemove], g_triangleData[indexFutureEnd]);
				cloneColor(g_colorData[indexFutureRemove], g_colorData[indexFutureEnd]);

				g_triangleCounts[strainFutureId]--;
			}
		}
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

extern "C" void launch_cudaFitness(dim3 grid, dim3 block, cudaArray* targetArray )
{
	//Bind texture
	cudaBindTextureToArray(texTarget, targetArray);
	struct cudaChannelFormatDesc desc;
	cudaGetChannelDesc(&desc, targetArray);

	//Run fitness function
	fitnessProcess<<< grid, block >>>();
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

struct uint2_sort_x
{
	__host__ __device__
	bool operator()(uint2 x, uint2 y)
	{
		return x.x < y.x;
	}
};

extern "C" uint2 getBestId(Settings &settings, void* fitnessData)
{
	//Sort fitness array
	int arraysize = settings.generationInfo.islandCount * settings.generationInfo.strainCount;

	try
	{
		thrust::device_ptr<uint2> dev_ptr((uint2*)fitnessData);
		thrust::sort(dev_ptr, dev_ptr+(arraysize-1), uint2_sort_x());
	}
	catch(thrust::system_error &e)
	{
		// output an error message and exit
		std::cerr  << e.what() << std::endl;
		exit(-1);
	}

	//Since the best Score as at the start of the block, copy first Uint2 to host and return it
	uint2 bestId;
	cudaMemcpy(&bestId, (uint2*)fitnessData, sizeof(uint2), cudaMemcpyDeviceToHost);

	return bestId;
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