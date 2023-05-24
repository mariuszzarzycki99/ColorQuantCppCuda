#include "kernel.h"
#include "device_launch_parameters.h"

#include <stdio.h>

void InfoCuda()
{
	int nDevices;
	cudaGetDeviceCount(&nDevices);

	printf("Number of devices: %d\n", nDevices);

	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("  Total global memory (Gbytes) %.1f\n", (float)(prop.totalGlobalMem) / 1024.0 / 1024.0 / 1024.0);
		printf("  Shared memory per block (Kbytes) %.1f\n", (float)(prop.sharedMemPerBlock) / 1024.0);
		printf("  Warp-size: %d\n", prop.warpSize);
		printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
		printf("  Concurrent computation/communication: %s\n", prop.deviceOverlap ? "yes" : "no");
		printf("  MaxBlocksPerMultiProcessor: %d\n", prop.maxBlocksPerMultiProcessor);
		printf("  MaxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
		printf("  MultiProcessorCount: %d\n", prop.multiProcessorCount);
		printf("  MaxThreadsPerMultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);
	}
}


