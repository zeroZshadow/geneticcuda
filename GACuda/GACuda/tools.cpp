#include "StdAfx.h"
#include "tools.h"

#include <stdio.h>

GLuint tools::loadTexture(char* aFilename)
{

	//Load terrain textures
	til::Image* image;
	GLuint textureId;
	image = til::TIL_Load(aFilename, TIL_FILE_ABSOLUTEPATH | TIL_DEPTH_A8B8G8R8);	//Side

	glGenTextures(1, &textureId);

	if(image == 0 || textureId == 0)
	{
		printf("> Could not load texture\n");
		return 0;
	}

	glBindTexture(GL_TEXTURE_2D, textureId);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexImage2D(
		GL_TEXTURE_2D, 0, GL_RGBA,
		image->GetPitchX(), image->GetPitchY(),
		0,
		GL_RGBA, GL_UNSIGNED_BYTE, image->GetPixels()
		);
	glBindTexture(GL_TEXTURE_2D, 0);

	til::TIL_Release(image);
	return textureId;
}