#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#pragma warning(disable:4996)
#endif

//OpenGL
#include <GL/glew.h>
#include <GL/freeglut.h>

//TIL
#include "TinyImageLoader.h"