#include "StdAfx.h"
#include "settings.h"

settings* settings::instance = 0;
MutationRates settings::mutationRates;
MutationRanges settings::mutationRanges;
ImageInfo settings::imageInfo;
GenerationInfo settings::generationInfo;

void settings::init()
{
	if (instance) return;
	instance = new settings();
	instance->initInstance(0, 0, 0);
}

//TODO UPDATE
void settings::init( MutationRates* rates, MutationRanges* ranges, ImageInfo* info )
{
	if (instance) return;
	instance = new settings();
	instance->initInstance(rates, ranges, info);
}

//TODO UPDATE
void settings::initInstance( MutationRates* rates, MutationRanges* ranges, ImageInfo* info )
{
	if (!(rates && ranges && info))
	{
		reset();
		return;
	}

	//Set values;
	imageInfo = *info;
	mutationRates = *rates;
	mutationRanges = *ranges;
}

void settings::reset()
{
	//Image info
	imageInfo.imageWidth = 256;
	imageInfo.imageHeight = 256;

	//Colors
	mutationRates.alphaMutationRate	= 1500;
	mutationRanges.alphaRangeMin	= 30;
	mutationRanges.alphaRangeMax	= 60;
	mutationRates.redMutationRate	= 1500;
	mutationRanges.redRangeMin		= 0;
	mutationRanges.redRangeMax		= 255;
	mutationRates.greenMutationRate	= 1500;
	mutationRanges.greenRangeMin	= 0;
	mutationRanges.greenRangeMax	= 255;
	mutationRates.blueMutationRate	= 1500;
	mutationRanges.blueRangeMin		= 0;
	mutationRanges.blueRangeMax		= 255;

	//Point
	mutationRates.pointMaxMoveMutationRate	= 1500;
	mutationRates.pointMidMoveMutationRate	= 1500;
	mutationRates.pointMinMoveMutationRate	= 1500;
	mutationRanges.pointMidMoveRange		= 20;
	mutationRanges.pointMinMoveRange		= 3;

	//Strain
	mutationRates.strainAddTriangleMutationRate		= 700;
	mutationRates.strainRemoveTriangleMutationRate	= 1500;
	mutationRates.strainMoveTriangleMutationRate	= 500;
	mutationRanges.strainMinTriangles = 1;
	mutationRanges.strainMaxTriangles = 255;

	//
	generationInfo.generationSize = 32;
}
