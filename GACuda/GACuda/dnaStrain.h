#pragma once

#include "dnaTriangle.h"

struct dnaStrain
{
	unsigned int m_triangleCount;
	dnaTriangle* m_triangles;
};

void initStrain(dnaStrain& self);
void cleanStrain(dnaStrain& self);
void cloneStrain(dnaStrain& self, dnaStrain& clone);
void mutateStrain(dnaStrain& self);
void printStrain(dnaStrain& self);

//mutations
void removeTriangle(dnaStrain& self);
void addTriangle(dnaStrain& self);