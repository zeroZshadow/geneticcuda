#include "StdAfx.h"
#include "dnaPoint.h"

#include "tools.h"
#include "settings.h"
#include <minmax.h>

void initPoint( dnaPoint& self )
{
	self.m_pos.x = tools::randomBetweenUINT(0, settings::imageInfo.imageWidth);
	self.m_pos.y = tools::randomBetweenUINT(0, settings::imageInfo.imageHeight);
}

void printPoint( dnaPoint& self, char* pad )
{
	printf("%sx:%u y:%u\n", pad, self.m_pos.x, self.m_pos.y);
}

void clonePoint ( dnaPoint& self, dnaPoint& clone )
{
	clone.m_pos = self.m_pos;
}

void mutatePoint( dnaPoint& self )
{

	if (tools::willMutate(settings::mutationRates.pointMaxMoveMutationRate))
	{
		self.m_pos.x = tools::randomBetween(0, settings::imageInfo.imageWidth);
		self.m_pos.y = tools::randomBetween(0, settings::imageInfo.imageHeight);
	}

	if (tools::willMutate(settings::mutationRates.pointMidMoveMutationRate))
	{
		self.m_pos.x = min( max(0, self.m_pos.x + tools::randomBetween(-settings::mutationRanges.pointMidMoveRange, settings::mutationRanges.pointMidMoveRange)), settings::imageInfo.imageWidth);
		self.m_pos.y = min( max(0, self.m_pos.y + tools::randomBetween(-settings::mutationRanges.pointMidMoveRange, settings::mutationRanges.pointMidMoveRange)), settings::imageInfo.imageHeight);
	}

	if (tools::willMutate(settings::mutationRates.pointMinMoveMutationRate))
	{
		self.m_pos.x = min( max(0, self.m_pos.x + tools::randomBetween(-settings::mutationRanges.pointMinMoveRange, settings::mutationRanges.pointMinMoveRange)), settings::imageInfo.imageWidth);
		self.m_pos.y = min( max(0, self.m_pos.y + tools::randomBetween(-settings::mutationRanges.pointMinMoveRange, settings::mutationRanges.pointMinMoveRange)), settings::imageInfo.imageHeight);
	}
}