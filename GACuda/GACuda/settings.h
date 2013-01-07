#pragma once

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
	UINT stringAddTriangleMutationRate;
	UINT stringRemoveTriangleMutationRate;
	UINT stringMoveTriangleMutationRate;
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
	UINT pointMidMoveRange;
	UINT pointMinMoveRange;

	//String
	UINT stringMinTriangles;
	UINT stringMaxTriangles;
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
private:
	void initInstance(MutationRates* rates, MutationRanges* ranges, ImageInfo* info);
	void reset();

	static settings* instance;
};

