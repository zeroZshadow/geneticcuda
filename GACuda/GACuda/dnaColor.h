#pragma once
class dnaColor
{
public:
	dnaColor(void);

	void init();
	void clone(dnaColor* color);
	void mutate();
	void print(char* pad = "");

	union{
		uint1 m_color;
		uchar4 m_components;
	};
};

