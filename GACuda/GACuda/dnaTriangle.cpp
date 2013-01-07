#include "StdAfx.h"
#include "dnaTriangle.h"

#include "tools.h"
#include "settings.h"
#include <minmax.h>


dnaTriangle::dnaTriangle(void)
{
	for(int i = 0; i < 3; ++i)
	{
		m_points[i].m_pos.x = 0;
		m_points[i].m_pos.y = 0;
	}
}

void dnaTriangle::init()
{
	dnaPoint origin; //throwaway point as triangle origin
	origin.init();

	for(int i = 0; i < 3; ++i)
	{
		m_points[i].m_pos.x = min(max(0, origin.m_pos.x + tools::randomBetween(-3, 3)), settings::imageInfo.imageWidth);
		m_points[i].m_pos.y = min(max(0, origin.m_pos.y + tools::randomBetween(-3, 3)), settings::imageInfo.imageHeight);
	}
	
	m_color.init();
}

void dnaTriangle::print(char* pad)
{
	char buffer[16]; sprintf_s(buffer, "%s ", pad);
	//Points
	printf("%sm_points {\n", pad);
	for(int i = 0; i < 3; ++i)
	{
		m_points[i].print(buffer);
	}
	printf("%s}\n", pad);

	//Color
	printf("%sm_color {\n", pad);
	m_color.print(buffer);
	printf("%s}\n", pad);
}

void dnaTriangle::clone(dnaTriangle* triangle)
{

}

void dnaTriangle::mutate()
{
	m_color.mutate();
	for(int i=0; i<3;++i)
		m_points[i].mutate();
}