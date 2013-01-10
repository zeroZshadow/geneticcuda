#pragma once
#include "settings.h"

class framework
{
public:
	framework(void);
	~framework(void);

	void process();
	void initialize();
	void setupRNG(Settings& settings);

	void renderScene();
	void allocateGMem();
	void createBestTexture();

private:
	GLuint m_TargetTexture;
	GLuint m_BestTexture;

	cudaGraphicsResource* m_cudaTargetTexture;
	cudaGraphicsResource* m_cudaBestTexture;

	void* m_cudaTriangles;
	void* m_cudaColors;
	void* m_cudaTriangleCounts;
	void* m_cudaDrawBuffer;
	void* m_cudaBestBuffer;
	curandState* m_cudaRandState;
	void* m_cudaRasterLines;
	void* m_cudaFitness;

	int m_drawBufferSize;
	unsigned int m_BestStrain;

	Settings m_settings;
};

