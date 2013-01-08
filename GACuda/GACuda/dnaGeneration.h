#pragma once

#include "dnaStrain.h"

struct dnaGeneration
{
	dnaStrain* m_strains;
};

void initGeneration(dnaGeneration& self);
void cleanGeneration(dnaGeneration& self);
void cloneGeneration(dnaGeneration& self, dnaGeneration& clone);

/*class dnaGeneration
{
public:
	dnaGeneration(void);
	~dnaGeneration(void);

	void init();
	void clone(dnaGeneration& generation);

	dnaStrain* m_strains;
};*/

