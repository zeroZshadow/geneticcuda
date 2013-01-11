#pragma once

struct GenerationInfo
{
	int strainCount;
	int islandCount;
};

struct MutationRates
{
	//mutation ranges
	//Color
	int alphaMutationRate;
	int redMutationRate;
	int greenMutationRate;
	int blueMutationRate;

	//Point
	int pointMaxMoveMutationRate;
	int pointMidMoveMutationRate;
	int pointMinMoveMutationRate;

	//String
	int strainAddTriangleMutationRate;
	int strainRemoveTriangleMutationRate;
};

struct MutationRanges
{
	//Color
	int alphaRangeMin;
	int alphaRangeMax;
	int redRangeMin;
	int redRangeMax;
	int greenRangeMin;
	int greenRangeMax;
	int blueRangeMin;
	int blueRangeMax;

	//Point
	int pointMidMoveRange;
	int pointMinMoveRange;

	//String
	int strainMinTriangles;
	int strainMaxTriangles;
};

struct ImageInfo
{
	int imageWidth;
	int imageHeight;
};

struct Settings
{
	MutationRates mutationRates;
	MutationRanges mutationRanges;
	ImageInfo imageInfo;
	GenerationInfo generationInfo;
};

void initSettings(Settings& settings, unsigned int width, unsigned int height);