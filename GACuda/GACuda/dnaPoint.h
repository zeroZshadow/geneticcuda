#pragma once
class dnaPoint
{
public:
	dnaPoint(void);

	void init();
	void clone(dnaPoint* p);
	void mutate();
	void print(char* pad = "");

	uint2 m_pos;
};

