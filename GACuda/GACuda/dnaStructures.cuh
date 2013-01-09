#ifndef __DNASTRUCTURES__
#define __DNASTRUCTURES__

struct Triangle
{
	int2 point[3];
};

struct Color
{
	union{
		unsigned int color;
		uchar4 components;
	};
};

#endif