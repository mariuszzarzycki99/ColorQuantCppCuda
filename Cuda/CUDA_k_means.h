#ifndef CUDA_K_MEANS_H
#define CUDA_K_MEANS_H

#include "cuda_runtime.h"
#include <inttypes.h>

#pragma pack(push,1)
typedef struct CudaPixel {
	unsigned char b, g, r;
}CudaPixel;
#pragma pack(pop)





float CUDA_K_means(int& colours, unsigned char* pixelTab, int size,bool reduction,int initMethod);
__host__ void cluster(int& colours, CudaPixel* pixelTab, int size,int initMethod);
__host__ void clusterWithoutReduction(int& colours, CudaPixel* pixelTab, int size, int initMethod);

#endif 