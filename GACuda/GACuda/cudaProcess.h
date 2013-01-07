#pragma once

extern "C" void
	launch_cudaProcess(dim3 grid, dim3 block, int sbytes);