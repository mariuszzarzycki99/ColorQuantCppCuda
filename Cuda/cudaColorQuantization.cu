#include "kernel.h"
#include "CUDA_median_cut.h"
#include "CUDA_k_means.h"

void kMeansCudaQuant(int& colours, unsigned char* pixelTab, int size, bool reduction, int initMethod)
{
	CUDA_K_means(colours, pixelTab, size, reduction, initMethod);
}
void medianCutCudaQuant(int& colours, uint8_t* pixelTab, int size)
{
	CUDA_medianCut(colours, pixelTab, size);
}
void getInfo()
{
	InfoCuda();
}