#include "StdAfx.h"

#include "framework.h"
#include "cudaProcess.h"

#include "tools.h"
#include "settings.h"
#include "dnaTriangle.h"

#include <stdio.h>

framework::framework(void)
{
	m_TargetTexture = 0;
	m_BestTexture = 0;

	m_cudaTriangleCounts = 0;
	m_cudaTriangles = 0;
	m_cudaColors = 0;
	m_cudaDrawBuffer = 0;
	m_cudaBestBuffer = 0;
	m_cudaRandState = 0;
	m_cudaRasterLines = 0;
	m_cudaFitness = 0;

	m_drawBufferSize = 0;
	m_BestStrain = make_uint2(UINT_MAX, 0);

	//Initialize textures
	m_TargetTexture = tools::loadTexture("./assets/test.png");
	CudaSafeCall(cudaGraphicsGLRegisterImage(&m_cudaTargetTexture, m_TargetTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));

	//Initialize settings
	initSettings(m_settings);

	//Initialize RNG
	setupRNG(m_settings);

	//Create best texture memory
	createBestTexture();

	//Allocate gpu memory blocks
	allocateGMem();

	//Upload constants to CUDA
	uploadConstants(
		m_settings,
		m_cudaRandState,
		m_cudaTriangleCounts,
		m_cudaTriangles,
		m_cudaColors,
		m_cudaBestBuffer,
		m_cudaDrawBuffer,
		m_cudaRasterLines,
		m_cudaFitness
	);

	//Initialize strains
	initialize();
}


framework::~framework(void)
{
	cudaFree(m_cudaTriangles);
	cudaFree(m_cudaColors);
	cudaFree(m_cudaTriangleCounts);
	cudaFree(m_cudaDrawBuffer);
	cudaFree(m_cudaBestBuffer);
	cudaFree(m_cudaRandState);
	cudaFree(m_cudaRasterLines);
	cudaFree(m_cudaFitness);
}

void framework::initialize()
{
	//Eval current Generations
	dim3 grid(m_settings.generationInfo.islandCount, 1, 1); //islands
	dim3 block(m_settings.generationInfo.strainCount, 1, 1); //generations per island

	printf( "Initialize RNG\n");
	launch_cudaSetupRNG(grid, block);
	CudaCheckError();

	printf("Initialize DNA\n");
	launch_cudaInitialize(grid, block);
	CudaCheckError();
}

void framework::process()
{
	//Clear draw buffers
	CudaSafeCall(cudaMemset(m_cudaDrawBuffer, 0, m_drawBufferSize));
	int num_texels = m_settings.imageInfo.imageWidth * m_settings.imageInfo.imageHeight;
	int size_tex_data = sizeof(UINT) * num_texels;

	//Eval current Generations
	dim3 grid(m_settings.generationInfo.islandCount, 1, 1); //islands
	dim3 block(m_settings.generationInfo.strainCount, 1, 1); //generations per island

	//Render strains in drawbuffer
	launch_cudaRender(grid, block);
	CudaCheckError();

	//Perform fitness function
	cudaArray *targetArrayPtr;
	CudaSafeCall(cudaGraphicsMapResources(1, &m_cudaTargetTexture));
	CudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&targetArrayPtr, m_cudaTargetTexture, 0, 0));
	launch_cudaFitness(grid, block, targetArrayPtr);
	CudaCheckError();
	CudaSafeCall(cudaGraphicsUnmapResources(1, &m_cudaTargetTexture));

	//WHAT IS THAT? THE IMAGE IS EVOLVING!!
	launch_cudaEvolve(grid, block);
	CudaCheckError();

	//DISPLAY BEST IMAGE
	uint2 bestStrain = getBestId(m_settings, m_cudaFitness);
	if (bestStrain.x < m_BestStrain.x) //Update stats when new best is better then overall best
	{
		m_BestStrain = bestStrain;
		printf("Best fitness %u\n", m_BestStrain.x);

		//Copy best from drawBuffer to bestBuffer
		UINT* imagebuffer = (UINT*)m_cudaDrawBuffer;
		CudaSafeCall(cudaMemcpy(m_cudaBestBuffer, &imagebuffer[m_BestStrain.y*num_texels], size_tex_data ,cudaMemcpyDeviceToDevice));

		//Update texture
		updateBestTexture();
	}
}

void framework::renderScene()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	//Draw here
	
	float width = (float)m_settings.imageInfo.imageWidth;
	float height = (float)m_settings.imageInfo.imageHeight;

	glEnable(GL_TEXTURE_2D);

	//Draw Target
	glBindTexture(GL_TEXTURE_2D, m_TargetTexture);
	glBegin(GL_QUADS);
	glTexCoord2f(0, 0); glVertex3f(0, 0, 0);
	glTexCoord2f(0, 1); glVertex3f(0, height, 0);
	glTexCoord2f(1, 1); glVertex3f(width, height, 0);
	glTexCoord2f(1, 0); glVertex3f(width, 0, 0);
	glEnd();

	//Draw current best

	glBindTexture(GL_TEXTURE_2D, m_BestTexture);
	glBegin(GL_QUADS);
	glTexCoord2f(0, 0); glVertex3f(256, 256, 0);
	glTexCoord2f(0, 1); glVertex3f(256, 256+height, 0);
	glTexCoord2f(1, 1); glVertex3f(256+width, 256+height, 0);
	glTexCoord2f(1, 0); glVertex3f(256+width, 256, 0);
	glEnd();

	//Unbind
	glBindTexture(GL_TEXTURE_2D, 0);
	
}

void framework::allocateGMem()
{
	int S = m_settings.generationInfo.islandCount * m_settings.generationInfo.strainCount;
	int TM = m_settings.mutationRanges.strainMaxTriangles;

	printf("allocating %i strains divided over %i islands\n", S, m_settings.generationInfo.islandCount);
	printf("each strain is %i triangles\n", TM);

	//Multiply data by 2 to hold older generations
	int countsize = sizeof(int) * S * 2;
	int trianglesize = sizeof(uint2) * 3 * TM * S * 2;
	int colorsize = sizeof(UINT) * TM * S * 2;
	int bestsize = sizeof(UINT) * m_settings.imageInfo.imageWidth * m_settings.imageInfo.imageHeight;
	int rasterlinesize = sizeof(int2) * S * m_settings.imageInfo.imageHeight;
	int fitnesssize = sizeof(uint2) * S;
	m_drawBufferSize = sizeof(UINT) * S * m_settings.imageInfo.imageWidth * m_settings.imageInfo.imageHeight;
	printf("Total size is %i bytes\n", countsize+trianglesize+colorsize+bestsize+m_drawBufferSize+rasterlinesize+fitnesssize);

	//Allocate block holding triangle counts
	CudaSafeCall(cudaMalloc((void**) &m_cudaTriangleCounts, countsize));

	//Allocate block holding triangle data
	CudaSafeCall(cudaMalloc((void**) &m_cudaTriangles, trianglesize));

	//Allocate block holding triangle colors
	CudaSafeCall(cudaMalloc((void**) &m_cudaColors, colorsize));

	//Allocate block for holding drawn strain data
	CudaSafeCall(cudaMalloc((void**) &m_cudaDrawBuffer, m_drawBufferSize));

	//Allocate block for holding best result texture
	CudaSafeCall(cudaMalloc((void**) &m_cudaBestBuffer, bestsize));
	CudaSafeCall(cudaMemset(m_cudaBestBuffer, 0, bestsize));
	updateBestTexture();

	//Allocate block for rasterizing triangles
	CudaSafeCall(cudaMalloc((void**) &m_cudaRasterLines, rasterlinesize));

	//Allocate block for holding fitness data per strain
	CudaSafeCall(cudaMalloc((void**) &m_cudaFitness, fitnesssize));
}

void framework::setupRNG(Settings& settings)
{
	int S = m_settings.generationInfo.islandCount * m_settings.generationInfo.strainCount;
	CudaSafeCall(cudaMalloc((void**) &m_cudaRandState, S*sizeof(curandState)));
}

void framework::createBestTexture()
{
	glGenTextures(1, &m_BestTexture);
	glBindTexture(GL_TEXTURE_2D, m_BestTexture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_settings.imageInfo.imageWidth, m_settings.imageInfo.imageHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	CudaSafeCall(cudaGraphicsGLRegisterImage(&m_cudaBestTexture, m_BestTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

	glBindTexture(GL_TEXTURE_2D, 0);
}

void framework::updateBestTexture()
{
	int num_texels = m_settings.imageInfo.imageWidth * m_settings.imageInfo.imageHeight;
	int size_tex_data = sizeof(UINT) * num_texels;

	//Copy bestTexture to texture
	cudaArray *arrayPtr;
	CudaSafeCall(cudaGraphicsMapResources(1, &m_cudaBestTexture));
	CudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&arrayPtr, m_cudaBestTexture, 0, 0));
	CudaSafeCall(cudaMemcpyToArray(arrayPtr, 0, 0, m_cudaBestBuffer, size_tex_data, cudaMemcpyDeviceToDevice));
	CudaSafeCall(cudaGraphicsUnmapResources(1, &m_cudaBestTexture));
}