#pragma once

struct GenerationInfo
{
	UINT generationSize;
};

struct MutationRates
{
	//mutation ranges
	//Color
	UINT alphaMutationRate;
	UINT redMutationRate;
	UINT greenMutationRate;
	UINT blueMutationRate;

	//Point
	UINT pointMaxMoveMutationRate;
	UINT pointMidMoveMutationRate;
	UINT pointMinMoveMutationRate;

	//String
	UINT strainAddTriangleMutationRate;
	UINT strainRemoveTriangleMutationRate;
	UINT strainMoveTriangleMutationRate;
};

struct MutationRanges
{
	//Color
	UINT alphaRangeMin;
	UINT alphaRangeMax;
	UINT redRangeMin;
	UINT redRangeMax;
	UINT greenRangeMin;
	UINT greenRangeMax;
	UINT blueRangeMin;
	UINT blueRangeMax;

	//Point
	int pointMidMoveRange;
	int pointMinMoveRange;

	//String
	UINT strainMinTriangles;
	UINT strainMaxTriangles;
};

struct ImageInfo
{
	UINT imageWidth;
	UINT imageHeight;
};

class settings
{
public:
	static void init();
	static void init(MutationRates* rates, MutationRanges* ranges, ImageInfo* info);

	static MutationRates mutationRates;
	static MutationRanges mutationRanges;
	static ImageInfo imageInfo;
	static GenerationInfo generationInfo;
private:
	void initInstance(MutationRates* rates, MutationRanges* ranges, ImageInfo* info);
	void reset();

	static settings* instance;
};

