#include "dnaGeneration.h"

__global__ void initProcess()
{

}

__global__ void fitnessProcess()
{

}

extern "C" void launch_fitnessProcess(dim3 grid, dim3 block)
{
	//dnaGeneration* generations = new dnaGeneration[grid.x];
	initProcess<<< grid, block >>>();
	fitnessProcess<<< grid, block >>>();
	//delete[] generations;
}