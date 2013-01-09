#pragma once

extern "C" void
	launch_cudaSetupRNG(dim3 grid, dim3 block);

extern "C" void
	launch_cudaInitialize(dim3 grid, dim3 block,
		void* counts, void* triangles, void* colors
	);
extern "C" void
	launch_cudaRender(dim3 grid, dim3 block,
		void* counts, void* triangles, void* colors, void* draw, void* raster
	);

extern "C" void
	launch_cudaFitness(dim3 grid, dim3 block,
		void* draw, void* best
	);

struct Settings;
extern "C" void
	uploadConstants(Settings& settings, curandState* randState);