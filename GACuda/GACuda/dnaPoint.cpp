#include "StdAfx.h"
#include "dnaPoint.h"

#include "tools.h"
#include "settings.h"

dnaPoint::dnaPoint(void)
{
	m_pos.x = 0;
	m_pos.y = 0;
}

void dnaPoint::init()
{
	m_pos.x = tools::randomBetween(0, settings::imageInfo.imageWidth);
	m_pos.y = tools::randomBetween(0, settings::imageInfo.imageHeight);
}

void dnaPoint::print( char* pad )
{
	printf("%sx:%u y:%u\n", pad, m_pos.x, m_pos.y);
}

void dnaPoint::clone(dnaPoint* p)
{
	p->m_pos = m_pos;
}

void dnaPoint::mutate()
{
	/*
	if (Tools.WillMutate(Settings.ActiveMovePointMaxMutationRate))
	{
	X = Tools.GetRandomNumber(0, Tools.MaxWidth);
	Y = Tools.GetRandomNumber(0, Tools.MaxHeight);
	drawing.SetDirty();
	}

	if (Tools.WillMutate(Settings.ActiveMovePointMidMutationRate))
	{
	X =
	Math.Min(
	Math.Max(0,
	X +
	Tools.GetRandomNumber(-Settings.ActiveMovePointRangeMid,
	Settings.ActiveMovePointRangeMid)), Tools.MaxWidth);
	Y =
	Math.Min(
	Math.Max(0,
	Y +
	Tools.GetRandomNumber(-Settings.ActiveMovePointRangeMid,
	Settings.ActiveMovePointRangeMid)), Tools.MaxHeight);
	drawing.SetDirty();
	}

	if (Tools.WillMutate(Settings.ActiveMovePointMinMutationRate))
	{
	X =
	Math.Min(
	Math.Max(0,
	X +
	Tools.GetRandomNumber(-Settings.ActiveMovePointRangeMin,
	Settings.ActiveMovePointRangeMin)), Tools.MaxWidth);
	Y =
	Math.Min(
	Math.Max(0,
	Y +
	Tools.GetRandomNumber(-Settings.ActiveMovePointRangeMin,
	Settings.ActiveMovePointRangeMin)), Tools.MaxHeight);
	drawing.SetDirty();
	}
	*/
}