#include "StdAfx.h"
#include "dnaString.h"
#include "dnaTriangle.h"

#include "tools.h"
#include "settings.h"

dnaString::dnaString(void)
{
	m_triangleCount = 0;
}

dnaString::~dnaString(void)
{
	clean();
}

void dnaString::clean()
{
	m_triangles.clear();
}

void dnaString::init()
{
	m_triangleCount = 0;
	m_triangles.clear();
	for( UINT i = 0; i < settings::mutationRanges.stringMinTriangles; ++i)
	{
		addTriangle();
	}
}

void dnaString::print()
{
	printf("stringDna {\n");
	printf(" m_triangleCount = %u\n", m_triangleCount);

	//Print Triangles
	printf(" m_triangles {\n");

	for( UINT i = 0; i < m_triangleCount; ++i)
	{
		m_triangles[i].print("  ");
	}

	printf(" }\n");

	printf("}\n");
}

void dnaString::clone(dnaString* dna)
{
	dna->clean();
	dna->m_triangleCount = m_triangleCount;
	dna->m_triangles = m_triangles;
}

void dnaString::mutate()
{
	if (tools::willMutate(settings::mutationRates.stringAddTriangleMutationRate))
		addTriangle();

	if (tools::willMutate(settings::mutationRates.stringRemoveTriangleMutationRate))
		removeTriangle();

	if (tools::willMutate(settings::mutationRates.stringMoveTriangleMutationRate))
		moveTriangle();

	for( UINT i = 0; i < m_triangleCount; ++i)
	{
		m_triangles[i].mutate();
	}
}

void dnaString::moveTriangle()
{
	if (m_triangleCount < 1)
		return;

	std::vector<dnaTriangle>::iterator itremove = m_triangles.begin();
	const int removeoffset = tools::randomBetween(0, m_triangleCount);
	itremove += removeoffset;
	dnaTriangle t = *itremove;
	m_triangles.erase(itremove);

	std::vector<dnaTriangle>::iterator itinsert = m_triangles.begin();
	const int insertoffset = tools::randomBetween(0, m_triangleCount);
	itinsert += insertoffset;
	m_triangles.insert(itinsert, t);
}

void dnaString::removeTriangle()
{
	if (m_triangleCount > settings::mutationRanges.stringMinTriangles)
	{
		std::vector<dnaTriangle>::iterator it = m_triangles.begin();
		const int offset = tools::randomBetween(0, m_triangleCount);
		it += offset;
		m_triangles.erase(it);

		m_triangleCount--;
	}
}

void dnaString::addTriangle()
{
	if (m_triangleCount < settings::mutationRanges.stringMaxTriangles)
	{
		dnaTriangle t;
		t.init();

		std::vector<dnaTriangle>::iterator it = m_triangles.begin();
		const int offset = tools::randomBetween(0, m_triangleCount);
		it += offset;
		m_triangles.insert(it, t);

		m_triangleCount++;
	}
}