#ifndef CUDA_MEDIAN_CUT_H
#define CUDA_MEDIAN_CUT_H
#include "cuda_runtime.h"
#include <inttypes.h>
#include <vector>

typedef struct PixelBucket {
	unsigned int start, end;
}PixelBucket;

typedef struct NewColor {
	unsigned char b, g, r;
}NewColor;

float CUDA_medianCut(int& colours, uint8_t* pixelTab, int size);
__host__ void quantize(int& colours, uint8_t* pixelTab, int size);


#endif 