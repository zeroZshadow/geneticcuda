#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#pragma warning(disable:4996)
#endif

//OpenGL
#include <GL/glew.h>
#include <GL/freeglut.h>

//CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

//TIL
#include "TinyImageLoader.h"

//STD
#include <vector>
#include <stdio.h>

#include "ErrorCheck.h"