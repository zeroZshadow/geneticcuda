#include "StdAfx.h"
#include "dnaColor.h"

#include "tools.h"
#include "settings.h"


dnaColor::dnaColor(void)
{
	m_components.x = 0;
	m_components.y = 0;
	m_components.z = 0;
	m_components.w = 0;
}

void dnaColor::init()
{
	m_components.x = tools::randomBetween(settings::mutationRanges.redRangeMin,		settings::mutationRanges.redRangeMax);
	m_components.y = tools::randomBetween(settings::mutationRanges.greenRangeMin,	settings::mutationRanges.greenRangeMax);
	m_components.z = tools::randomBetween(settings::mutationRanges.blueRangeMin,	settings::mutationRanges.blueRangeMax);
	m_components.w = tools::randomBetween(settings::mutationRanges.alphaRangeMin,	settings::mutationRanges.alphaRangeMax);
}

void dnaColor::print( char* pad )
{
	printf("%s0x%08X\n", pad, m_color);
	printf("%sr:%u g:%u b:%u a:%u\n", pad, m_components.x, m_components.y, m_components.z, m_components.w);
}

void dnaColor::clone( dnaColor* color )
{
	color->m_color = m_color;
}

void dnaColor::mutate()
{
	/*
	if (Tools.WillMutate(Settings.ActiveRedMutationRate))
	{
		Red = Tools.GetRandomNumber(Settings.ActiveRedRangeMin, Settings.ActiveRedRangeMax);
	}

	if (Tools.WillMutate(Settings.ActiveGreenMutationRate))
	{
		Green = Tools.GetRandomNumber(Settings.ActiveGreenRangeMin, Settings.ActiveGreenRangeMax);
	}

	if (Tools.WillMutate(Settings.ActiveBlueMutationRate))
	{
		Blue = Tools.GetRandomNumber(Settings.ActiveBlueRangeMin, Settings.ActiveBlueRangeMax);
	}

	if (Tools.WillMutate(Settings.ActiveAlphaMutationRate))
	{
		Alpha = Tools.GetRandomNumber(Settings.ActiveAlphaRangeMin, Settings.ActiveAlphaRangeMax);
	}
	*/
}