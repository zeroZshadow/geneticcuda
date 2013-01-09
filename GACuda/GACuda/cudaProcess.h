#pragma once

extern "C" void
	launch_cudaInitialize(dim3 grid, dim3 block,
		void* counts, void* triangles, void* colors
	);
extern "C" void
	launch_cudaRender(dim3 grid, dim3 block,
		void* counts, void* triangles, void* colors, void* draw
	);

extern "C" void
	launch_cudaProcess(dim3 grid, dim3 block,
		void* draw
	);

struct Settings;
extern "C" void
	uploadSettings(Settings& settings);