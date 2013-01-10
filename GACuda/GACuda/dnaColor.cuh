#ifndef __DNACOLOR__
#define __DNACOLOR__

__device__ void initColor(Color& colorData, Settings& settings)
{
	colorData.components.x = randomBetween(settings.mutationRanges.redRangeMin,		settings.mutationRanges.redRangeMax);
	colorData.components.y = randomBetween(settings.mutationRanges.greenRangeMin,	settings.mutationRanges.greenRangeMax);
	colorData.components.z = randomBetween(settings.mutationRanges.blueRangeMin,	settings.mutationRanges.blueRangeMax);
	colorData.components.w = randomBetween(settings.mutationRanges.alphaRangeMin,	settings.mutationRanges.alphaRangeMax);
}

__device__ inline void cloneColor(Color& colorDst, const Color& colorSrc)
{
	colorDst.color = colorSrc.color;
}

#endif