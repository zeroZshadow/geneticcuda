#include "StdAfx.h"
#include "settings.h"

void initSettings(Settings& settings, unsigned int width, unsigned int height)
{
	//Image info
	settings.imageInfo.imageWidth	= width;
	settings.imageInfo.imageHeight	= height;

	//Colors
	settings.mutationRates.alphaMutationRate	= 500;
	settings.mutationRanges.alphaRangeMin		= 30;
	settings.mutationRanges.alphaRangeMax		= 60;
	settings.mutationRates.redMutationRate		= 500;
	settings.mutationRanges.redRangeMin			= 0;
	settings.mutationRanges.redRangeMax			= 255;
	settings.mutationRates.greenMutationRate	= 500;
	settings.mutationRanges.greenRangeMin		= 0;
	settings.mutationRanges.greenRangeMax		= 255;
	settings.mutationRates.blueMutationRate		= 500;
	settings.mutationRanges.blueRangeMin		= 0;
	settings.mutationRanges.blueRangeMax		= 255;

	//Point
	settings.mutationRates.pointMaxMoveMutationRate	= 500;
	settings.mutationRates.pointMidMoveMutationRate	= 500;
	settings.mutationRates.pointMinMoveMutationRate	= 500;
	settings.mutationRanges.pointMidMoveRange		= 20;
	settings.mutationRanges.pointMinMoveRange		= 3;

	//Strain
	settings.mutationRates.strainAddTriangleMutationRate	= 300;
	settings.mutationRates.strainRemoveTriangleMutationRate	= 500;
	settings.mutationRanges.strainMinTriangles = 1;
	settings.mutationRanges.strainMaxTriangles = 128;

	//Generation & Island
	settings.generationInfo.strainCount = 32;
	settings.generationInfo.islandCount = 1;
}