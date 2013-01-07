#include <stdio.h>

__global__ void
cudaProcess()
{
	//Grid == island
	//block == DNA
}

extern "C" void
launch_cudaProcess(dim3 grid, dim3 block, int sbytes)
{
        cudaProcess<<< grid, block, sbytes >>>();
}