#include "StdAfx.h"
#include "framework.h"
#include "cudaProcess.h"

#include "tools.h"
#include "settings.h"

#include <stdio.h>

framework::framework(void)
{
	m_TargetTexture = 0;
	m_BestTexture = 0;

	m_TargetTexture = tools::loadTexture("./assets/test.png");
	m_BestTexture = m_TargetTexture; //TEMP

	settings::init();
}


framework::~framework(void)
{
}

void framework::renderScene()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	//Draw here
	
	float width = (float)settings::imageInfo.imageWidth;
	float height = (float)settings::imageInfo.imageHeight;

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

void framework::process()
{
	//Eval current Generations
	dim3 grid(32, 1, 1); //islands
	dim3 block(settings::generationInfo.generationSize, 1, 1); //generations

	launch_fitnessProcess(grid, block);
	cudaError e = cudaGetLastError();
	if (e != cudaSuccess)
	{
		printf("ERROR\n");
	}

	//New generation
	//Selection

	//Mutation
}