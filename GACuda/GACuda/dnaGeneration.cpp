#include "StdAfx.h"
#include "dnaGeneration.h"

#include "settings.h"

void initGeneration( dnaGeneration& self )
{
	self.m_strains = new dnaStrain[settings::generationInfo.generationSize];
}

void cleanGeneration( dnaGeneration& self )
{
	delete[] self.m_strains;
	self.m_strains = 0;
}

void cloneGeneration( dnaGeneration& self, dnaGeneration& clone )
{
	for( int i = 0; i < settings::generationInfo.generationSize; ++i)
	{
		cloneStrain(self.m_strains[i], clone.m_strains[i]);
	}
}