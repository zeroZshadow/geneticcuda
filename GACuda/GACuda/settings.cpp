#include "StdAfx.h"
#include "settings.h"

void initSettings(Settings& settings)
{
	//Image info
	settings.imageInfo.imageWidth	= 256;
	settings.imageInfo.imageHeight	= 256;

	//Colors
	settings.mutationRates.alphaMutationRate	= 1500;
	settings.mutationRanges.alphaRangeMin		= 30;
	settings.mutationRanges.alphaRangeMax		= 60;
	settings.mutationRates.redMutationRate		= 1500;
	settings.mutationRanges.redRangeMin			= 0;
	settings.mutationRanges.redRangeMax			= 255;
	settings.mutationRates.greenMutationRate	= 1500;
	settings.mutationRanges.greenRangeMin		= 0;
	settings.mutationRanges.greenRangeMax		= 255;
	settings.mutationRates.blueMutationRate		= 1500;
	settings.mutationRanges.blueRangeMin		= 0;
	settings.mutationRanges.blueRangeMax		= 255;

	//Point
	settings.mutationRates.pointMaxMoveMutationRate	= 1500;
	settings.mutationRates.pointMidMoveMutationRate	= 1500;
	settings.mutationRates.pointMinMoveMutationRate	= 1500;
	settings.mutationRanges.pointMidMoveRange		= 20;
	settings.mutationRanges.pointMinMoveRange		= 3;

	//Strain
	settings.mutationRates.strainAddTriangleMutationRate	= 700;
	settings.mutationRates.strainRemoveTriangleMutationRate	= 1500;
	settings.mutationRates.strainMoveTriangleMutationRate	= 500;
	settings.mutationRanges.strainMinTriangles = 1;
	settings.mutationRanges.strainMaxTriangles = 256;

	//Generation & Island
	settings.generationInfo.strainCount = 32;
	settings.generationInfo.islandCount = 8;
}