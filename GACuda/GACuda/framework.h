#pragma once
#include "settings.h"

class framework
{
public:
	framework(char* filename);
	~framework(void);

	void process();
	void initialize();
	void setupRNG(Settings& settings);

	void renderScene();
	void allocateGMem();
	void loadTargetImage(char* filename);
	void createBestTexture();
	void updateBestTexture();

private:
	GLuint m_TargetTexture;
	GLuint m_BestTexture;

	GLuint m_TargetWidth;
	GLuint m_TargetHeight;

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
	uint2 m_BestStrain;

	Settings m_settings;
};

