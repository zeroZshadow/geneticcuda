#include "StdAfx.h"

/*
#include "dnaColor.h"

#include "tools.h"
#include "settings.h"

void initColor(dnaColor& self)
{
	self.m_components.x = tools::randomBetween(settings.mutationRanges.redRangeMin,	settings.mutationRanges.redRangeMax);
	self.m_components.y = tools::randomBetween(settings.mutationRanges.greenRangeMin,	settings.mutationRanges.greenRangeMax);
	self.m_components.z = tools::randomBetween(settings.mutationRanges.blueRangeMin,	settings.mutationRanges.blueRangeMax);
	self.m_components.w = tools::randomBetween(settings.mutationRanges.alphaRangeMin,	settings.mutationRanges.alphaRangeMax);
}

void printColor(dnaColor& self, char* pad )
{
	printf("%s0x%08X\n", pad, self.m_color);
	uchar4& m_components = self.m_components;
	printf("%sr:%u g:%u b:%u a:%u\n", pad, m_components.x, m_components.y, m_components.z, m_components.w);
}

void cloneColor(dnaColor& self, dnaColor& clone )
{
	clone.m_color = self.m_color;
}

void mutateColor(dnaColor& self)
{
	uchar4& m_components = self.m_components;
	if (tools::willMutate(settings.mutationRates.redMutationRate))
	{
		m_components.x = tools::randomBetween(settings.mutationRanges.redRangeMin, settings.mutationRanges.redRangeMax);
	}

	if (tools::willMutate(settings.mutationRates.greenMutationRate))
	{
		m_components.y = tools::randomBetween(settings.mutationRanges.greenRangeMin, settings.mutationRanges.greenRangeMax);
	}

	if (tools::willMutate(settings.mutationRates.blueMutationRate))
	{
		m_components.z = tools::randomBetween(settings.mutationRanges.blueRangeMin, settings.mutationRanges.blueRangeMax);
	}

	if (tools::willMutate(settings.mutationRates.alphaMutationRate))
	{
		 m_components.w = tools::randomBetween(settings.mutationRanges.alphaRangeMin, settings.mutationRanges.alphaRangeMax);
	}
}
*/