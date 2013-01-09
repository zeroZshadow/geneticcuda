#pragma once
#include "settings.h"

class framework
{
public:
	framework(void);
	~framework(void);

	void process();
	void initialize();

	void renderScene();
	void allocateGMem();

private:
	GLuint m_TargetTexture;
	GLuint m_BestTexture;

	cudaGraphicsResource* m_cudaTargetTexture;
	cudaGraphicsResource* m_cudaBestTexture;

	void* m_cudaTriangles;
	void* m_cudaColors;
	void* m_cudaTriangleCounts;
	void* m_cudaDrawBuffer;

	int m_drawBufferSize;

	Settings m_settings;
};

