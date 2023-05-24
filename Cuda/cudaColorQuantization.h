#ifndef CUDACOLORQUANTIZATION
#define CUDACOLORQUANTIZATION

void kMeansCudaQuant(int& colours, unsigned char* pixelTab, int size, bool reduction, int initMethod);
void medianCutCudaQuant(int& colours, uint8_t* pixelTab, int size);
void getInfo();

#endif

