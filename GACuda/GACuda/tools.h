#pragma once

class tools
{
public:
	static GLuint loadTexture(char* aFilename);
	static UINT nextRandom();
	static int randomBetween(int min, int max);
	static UINT randomBetweenUINT( UINT min, UINT max );
	static bool willMutate(UINT rate);
};

