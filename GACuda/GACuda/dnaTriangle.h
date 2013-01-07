#pragma once
#include "dnaPoint.h"
#include "dnaColor.h"

class dnaTriangle
{
public:
	dnaTriangle(void);

	void init();
	void clone(dnaTriangle* triangle);
	void mutate();
	void print(char* pad = "");

	dnaPoint m_points[3];
	dnaColor m_color;
};

