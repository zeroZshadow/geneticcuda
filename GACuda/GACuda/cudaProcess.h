#pragma once

extern "C" void
	launch_cudaSetupRNG( dim3 grid, dim3 block );

extern "C" void
	launch_cudaInitialize( dim3 grid, dim3 block);

extern "C" void
	launch_cudaRender(dim3 grid, dim3 block	);

extern "C" unsigned int
	launch_cudaFitness(dim3 grid, dim3 block, cudaArray* targetArray, void* fitness	);

extern "C" void
	launch_cudaEvolve(dim3 grid, dim3 block	);

struct Settings;
extern "C" void
	uploadConstants(Settings& settings,
		curandState* randState,
		void* counts,
		void* triangles,
		void* colors,
		void* best,
		void* draw,
		void* raster,
		void* fitness
	);