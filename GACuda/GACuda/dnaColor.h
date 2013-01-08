#pragma once

struct dnaColor
{
	union{
		uint1 m_color;
		uchar4 m_components;
	};
};

void initColor(dnaColor& self);
void cloneColor(dnaColor& self, dnaColor& clone);
void mutateColor(dnaColor& self);
void printColor(dnaColor& self, char* pad = "");

/*class dnaColor
{
public:
	dnaColor(void);

	void init();
	void clone(dnaColor& color);
	void mutate();
	void print(char* pad = "");

	union{
		uint1 m_color;
		uchar4 m_components;
	};
};*/

