#include "StdAfx.h"

/*
#include "dnaStrain.h"
#include "dnaTriangle.h"

#include "tools.h"
#include "settings.h"

void initStrain( dnaStrain& self )
{
	self.m_triangleCount = 0;
	self.m_triangles = new dnaTriangle[settings.mutationRanges.strainMaxTriangles];
	for( unsigned int i = 0; i < settings.mutationRanges.strainMinTriangles; ++i)
	{
		addTriangle(self);
	}
}

void cleanStrain( dnaStrain& self )
{
	delete[] self.m_triangles;
	self.m_triangles = 0;
	self.m_triangleCount = 0;
}

void printStrain( dnaStrain& self )
{
	printf("stringDna {\n");
	printf(" m_triangleCount = %u\n", self.m_triangleCount);

	//Print Triangles
	printf(" m_triangles {\n");

	for( unsigned int i = 0; i < self.m_triangleCount; ++i)
	{
		printTriangle(self.m_triangles[i], "  ");
	}

	printf(" }\n");
	printf("}\n");
}

void cloneStrain( dnaStrain& self, dnaStrain& clone )
{
	clone.m_triangleCount = self.m_triangleCount;
	for( unsigned int i = 0; i < self.m_triangleCount; ++i)
		cloneTriangle(self.m_triangles[i], clone.m_triangles[i]);
}

void mutateStrain( dnaStrain& self )
{
	if (tools::willMutate(settings.mutationRates.strainAddTriangleMutationRate))
		addTriangle(self);

	if (tools::willMutate(settings.mutationRates.strainRemoveTriangleMutationRate))
		removeTriangle(self);

	for( unsigned int i = 0; i < self.m_triangleCount; ++i)
	{
		mutateTriangle(self.m_triangles[i]);
	}
}

void removeTriangle( dnaStrain& self )
{
	if (self.m_triangleCount > settings.mutationRanges.strainMinTriangles)
	{
		const unsigned int offset = tools::randomBetweenUINT(0, self.m_triangleCount); //Exclusive

		//Overwrite triangle to delete with last triangle
		cloneTriangle(self.m_triangles[self.m_triangleCount-1], self.m_triangles[offset]);

		self.m_triangleCount--;
	}
}

void addTriangle( dnaStrain& self )
{
	if (self.m_triangleCount < settings.mutationRanges.strainMaxTriangles)
	{
		//TODO: Always add at the end ? if drawing is additive
		const unsigned int offset = tools::randomBetweenUINT(0, self.m_triangleCount); //Exclusive

		//Move triangle to overwrite to the back
		cloneTriangle(self.m_triangles[offset], self.m_triangles[self.m_triangleCount]);
		//Overwrite original
		initTriangle(self.m_triangles[offset]);

		self.m_triangleCount++;
	}
}*/