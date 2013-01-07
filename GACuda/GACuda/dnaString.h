#pragma once

class dnaTriangle;
class dnaString
{
public:
	dnaString(void);
	~dnaString(void);

	void clean();

	void init();
	void clone(dnaString* dna);
	void mutate();
	void print();

	//mutations
	void moveTriangle();
	void removeTriangle();
	void addTriangle();

	UINT m_triangleCount;
	std::vector<dnaTriangle> m_triangles;
};

