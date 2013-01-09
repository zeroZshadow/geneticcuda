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

	m_cudaTriangles = 0;
	m_cudaColors = 0;
	m_cudaTriangleCounts = 0;
	m_cudaDrawBuffer = 0;

	m_drawBufferSize = 0;

	//Initialize textures
	m_TargetTexture = tools::loadTexture("./assets/test.png");
	m_BestTexture = tools::loadTexture("./assets/test.png");

	CudaSafeCall(cudaGraphicsGLRegisterImage(&m_cudaTargetTexture, m_TargetTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
	CudaSafeCall(cudaGraphicsGLRegisterImage(&m_cudaBestTexture, m_BestTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));

	//Initialize settings and upload to the GPU
	initSettings(m_settings);
	uploadSettings(m_settings);

	//Allocate gpu memory blocks
	allocateGMem();

	//Initialize strains
	initialize();
}


framework::~framework(void)
{
	cudaFree(m_cudaTriangles);
	cudaFree(m_cudaColors);
	cudaFree(m_cudaTriangleCounts);
	cudaFree(m_cudaDrawBuffer);
}

void framework::initialize()
{
	//Eval current Generations
	dim3 grid(m_settings.generationInfo.islandCount, 1, 1); //islands
	dim3 block(m_settings.generationInfo.strainCount, 1, 1); //generations per island

	printf("Initialize DNA\n");
	launch_cudaInitialize(grid, block, m_cudaTriangleCounts, m_cudaTriangles, m_cudaColors);
	CudaCheckError();
}

void framework::process()
{
	//Map target texture
	CudaSafeCall(cudaGraphicsMapResources(1, &m_cudaTargetTexture));
	CudaSafeCall(cudaGraphicsMapResources(1, &m_cudaBestTexture));

	//Eval current Generations
	dim3 grid(m_settings.generationInfo.islandCount, 1, 1); //islands
	dim3 block(m_settings.generationInfo.strainCount, 1, 1); //generations per island

	cudaMemset(m_cudaDrawBuffer, 0, m_drawBufferSize);

	launch_cudaRender(grid, block,
		m_cudaTriangleCounts, m_cudaTriangles, m_cudaColors, m_cudaDrawBuffer
	);
	CudaCheckError();

	//Unmap for rendering
	CudaSafeCall(cudaGraphicsUnmapResources(1, &m_cudaBestTexture));
	CudaSafeCall(cudaGraphicsUnmapResources(1, &m_cudaTargetTexture));
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

	printf("allocating %i strains (times 2)\n", S);
	printf("each strain is %i triangles\n", TM);

	int countsize = sizeof(int) * S * 2;
	int trianglesize = sizeof(uint2) * 3 * TM * S * 2;
	int colorsize = sizeof(UINT) * TM * S * 2;
	m_drawBufferSize = sizeof(UINT) * S * m_settings.imageInfo.imageWidth * m_settings.imageInfo.imageHeight;
	printf("Total size is %i bytes\n", countsize+trianglesize+colorsize+m_drawBufferSize);

	//Allocate block holding triangle counts
	CudaSafeCall(cudaMalloc((void**) &m_cudaTriangleCounts, countsize));

	//Allocate block holding triangle data
	CudaSafeCall(cudaMalloc((void**) &m_cudaTriangles, trianglesize));

	//Allocate block holding triangle colors
	CudaSafeCall(cudaMalloc((void**) &m_cudaColors, colorsize));

	//Allocate block for holding drawn strain data
	CudaSafeCall(cudaMalloc((void**) &m_cudaDrawBuffer, m_drawBufferSize));
}