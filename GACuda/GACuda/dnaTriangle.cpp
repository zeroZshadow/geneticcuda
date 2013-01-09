#include "StdAfx.h"

/*
#include "dnaTriangle.h"

#include "tools.h"
#include "settings.h"
#include <minmax.h>



void initTriangle( dnaTriangle& self )
{
	dnaPoint origin; //throwaway point as triangle origin
	initPoint(origin);

	for(int i = 0; i < 3; ++i)
	{
		self.m_points[i].m_pos.x = min(max(0, origin.m_pos.x + tools::randomBetween(-3, 3)), settings.imageInfo.imageWidth);
		self.m_points[i].m_pos.y = min(max(0, origin.m_pos.y + tools::randomBetween(-3, 3)), settings.imageInfo.imageHeight);
	}
	
	initColor(self.m_color);
}

void printTriangle( dnaTriangle& self, char* pad )
{
	char buffer[16]; sprintf_s(buffer, "%s ", pad);
	//Points
	printf("%sm_points {\n", pad);
	for(int i = 0; i < 3; ++i)
	{
		printPoint(self.m_points[i], buffer);
	}
	printf("%s}\n", pad);

	//Color
	printf("%sm_color {\n", pad);
	printColor(self.m_color, buffer);
	printf("%s}\n", pad);
}

void cloneTriangle( dnaTriangle& self, dnaTriangle& clone)
{
	cloneColor(self.m_color, clone.m_color);
	for(int i=0; i<3;++i)
		clonePoint(self.m_points[i], self.m_points[i]);
}

void mutateTriangle( dnaTriangle& self )
{
	mutateColor(self.m_color);
	for(int i=0; i<3;++i)
		mutatePoint(self.m_points[i]);
}*/