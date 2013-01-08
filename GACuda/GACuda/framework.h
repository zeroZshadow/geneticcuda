#pragma once
class framework
{
public:
	framework(void);
	~framework(void);

	void renderScene();
	void process();

private:
	GLuint m_TargetTexture;
	GLuint m_BestTexture;


};

