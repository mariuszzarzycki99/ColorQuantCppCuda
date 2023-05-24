#include "bitmap.hpp"
#include "k_means.hpp"
#include "median_cut.hpp"
#include "inputParser.h"
#include "programParams.h"

#include "../Cuda/cudaColorQuantization.h"

#include <stdio.h>
#include <chrono>
#include <string>
#include <iostream>

void kMeans(std::string image, int k, int maxIter, bool reduceData, int method, float oversampling);
void medianCut(std::string image, int k);
void kMeansCUDA(std::string image, int k, bool reduce, int method);
void medianCutCUDA(std::string image, int k);
void printCudaInfo();
int parseInput(int argc, char** argv, programParams &params);
void processImage(programParams& params);
void printHelp() { std::cout << "Avaiable commands:\nTODO\n"; }


int main(int argc, char** argv)
{
	programParams params;
	int result = parseInput(argc, argv, params);
	printCudaInfo();

	if (!result)
		processImage(params);
	else
		printHelp();

	return 0;
}

void kMeans(std::string image, int k, int maxIter, bool reduceData, int method, float oversampling)
{
	BMPImage bitmap;
	bitmap.ReadBMP(image);

	KMeans KMeans(bitmap.getPixelTab());
	float err = 1234.67f;
	auto start = std::chrono::high_resolution_clock::now();

	KMeans.ClusterData(k, method, err, maxIter, reduceData, oversampling);

	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout<<"kMeans: "<< (long long)duration.count()<<" ms\n";

	bitmap.SaveBMP("kmeans.bmp");
}
void medianCut(std::string image, int k)
{
	BMPImage bitmap;
	bitmap.ReadBMP(image);

	auto start = std::chrono::high_resolution_clock::now();

	MedianCut MedianCut(bitmap.getPixelTab());
	MedianCut.Quant(k);

	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

	std::cout << "medianCut: " << (long long)duration.count() << " ms\n";
	bitmap.SaveBMP("median_cut.bmp");
}
void kMeansCUDA(std::string image, int k, bool reduce, int method)
{
	BMPImage bitmap;
	bitmap.ReadBMP(image);

	auto start = std::chrono::high_resolution_clock::now();
	kMeansCudaQuant(k, (unsigned char*)bitmap.getPixelTab()->data(), bitmap.getPixelTabSize() * 3, reduce, method);
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "kMeansCUDA: " << (long long)duration.count() << " ms\n";
	bitmap.SaveBMP("k_means_cuda.bmp");
}
void medianCutCUDA(std::string image, int k)
{
	BMPImage bitmap;
	bitmap.ReadBMP(image);

	auto start = std::chrono::high_resolution_clock::now();
	medianCutCudaQuant(k, (unsigned char*)bitmap.getPixelTab()->data(), bitmap.getPixelTabSize() * 3);
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "medianCutCUDA: " << (long long)duration.count() << " ms\n";
	bitmap.SaveBMP("median_cut_cuda.bmp");
}
void printCudaInfo()
{
	getInfo();
}

int parseInput(int argc, char** argv,programParams &params)
{
	InputParser input(argc, argv);

	if (input.cmdOptionExists("-mc")) {
		params.bMedianCut = true;
	}
	if (input.cmdOptionExists("-kmeans")) {
		params.bKMeans = true;
	}
	if(!(params.bKMeans || params.bMedianCut))
	{
		printf("Wrong parameters\n");
		return -1;
	}
	if (input.cmdOptionExists("-cuda")) {
		params.cuda = true;
	}
	if (input.cmdOptionExists("-cpu")) {
		params.cpu = true;
	}
	if (input.cmdOptionExists("-colors")) {
		params.numColors = stoi(input.getCmdOption("-colors"));
	}
	if (input.cmdOptionExists("-f")) {
		const std::string& filename = input.getCmdOption("-f");
		if (!filename.empty()) {
			params.image = filename;
		}
	}
	if (input.cmdOptionExists("-iter")) {
		params.iterations = stoi(input.getCmdOption("-iter"));
	}
	if (input.cmdOptionExists("-reduce")) {
		params.reduce = true;
	}
	if (input.cmdOptionExists("-method")) {
		params.method = stoi(input.getCmdOption("-method"));
	}
	if (params.method == 2) {
		if (input.cmdOptionExists("-os")) {
			params.oversampling = stof(input.getCmdOption("-os"));
		}
	}
	return 0;
}

void processImage(programParams& params)
{
	if (params.bMedianCut)
	{
		if (params.cuda)
			medianCutCUDA(params.image, params.numColors);
		if(params.cpu)
			medianCut(params.image, params.numColors);
	}
	if (params.bKMeans)
	{
		if (params.cuda)
			kMeansCUDA(params.image, params.numColors, params.reduce, params.method);
		if(params.cpu)
			kMeans(params.image, params.numColors, params.iterations, params.reduce, params.method, params.oversampling);
	}
}