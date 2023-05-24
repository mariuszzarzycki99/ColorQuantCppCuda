#include "CUDA_k_means.h"
#include <device_launch_parameters.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <random>
#include <limits>

struct pixelLess : public thrust::binary_function<CudaPixel, CudaPixel, bool>
{
	__host__ __device__ bool operator()(const CudaPixel& first, const CudaPixel& second) const {
		if (first.b < second.b) return true;
		if (first.b > second.b) return false;

		if (first.g < second.g) return true;
		if (first.g > second.g) return false;

		if (first.r < second.r) return true;
		if (first.r > second.r) return false;

		return false;
	}
};
struct pixelEqual : public thrust::binary_function<CudaPixel, CudaPixel, bool>
{
	__host__ __device__ bool operator()(const CudaPixel& first, const CudaPixel& second) const {
		if (first.r == second.r && first.g == second.g && first.b == second.b) return true;
		return false;
	}
};
__device__ inline int pixelDistance2(unsigned char pixelBlue, unsigned char pixelGreen, unsigned char pixelRed, unsigned char centroidBlue, unsigned char centroidGreen, unsigned char centroidRed)
{
	return (pixelBlue - centroidBlue) * (pixelBlue - centroidBlue) + (pixelGreen - centroidGreen) * (pixelGreen - centroidGreen) + (pixelRed - centroidRed) * (pixelRed - centroidRed);
}
__global__ void assignNewValues(unsigned char* originalPixelTab, int pixelTabSize, unsigned char* centroidTab, int centroidSize)
{
	extern __shared__ unsigned char centroids[];
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	if (blockDim.x < centroidSize)
		for (int i = threadIdx.x, j = 0; j <= (centroidSize - 1) / blockDim.x; ++j, i += blockDim.x)
		{
			if (i < centroidSize)
				centroids[i] = centroidTab[i];
		}
	else if (threadIdx.x < centroidSize)
	{
		centroids[threadIdx.x] = centroidTab[threadIdx.x];
	}
	__syncthreads();
	if (index < pixelTabSize)
	{
		for (int i = index; i < pixelTabSize; i += stride)
		{
			int nearest = 0;
			int distance = 256 * 256 * 256;
			for (int j = 0; j < centroidSize / 3; j++)
			{
				int newDistance = pixelDistance2(originalPixelTab[3 * i], originalPixelTab[3 * i + 1], originalPixelTab[3 * i + 2], centroids[3 * j], centroids[3 * j + 1], centroids[3 * j + 2]);
				if (newDistance < distance)
				{
					distance = newDistance;
					nearest = j;
				}
			}
			originalPixelTab[3 * i] = centroids[3 * nearest];
			originalPixelTab[3 * i + 1] = centroids[3 * nearest + 1];
			originalPixelTab[3 * i + 2] = centroids[3 * nearest + 2];
		}
	}
}
__global__ void KMeansConvertToSOA(unsigned char* original, unsigned char* blueTab, unsigned char* greenTab, unsigned char* redTab, int size)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	if (index < size)
	{
		for (int i = index; i < size; i += stride)
		{
			blueTab[i] = original[3 * i];
			greenTab[i] = original[3 * i + 1];
			redTab[i] = original[3 * i + 2];
		}
	}
}
__global__ void reducedPixelsValueMultiply(unsigned char* blueTab, unsigned char* greenTab, unsigned char* redTab, int size, int* countTab, int* newBlueTab, int* newGreenTab, int* newRedTab)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	if (index < size)
	{
		for (int i = index; i < size; i += stride)
		{
			newBlueTab[i] = blueTab[i] * countTab[i];
			newGreenTab[i] = greenTab[i] * countTab[i];
			newRedTab[i] = redTab[i] * countTab[i];
		}
	}
}
bool checkCentroidDuplicate(thrust::host_vector<CudaPixel>& centroids, CudaPixel centroid)
{
	for (int i = 0; i < centroids.size(); ++i)
	{
		if (centroids[i].b == centroid.b && centroids[i].g == centroid.g && centroids[i].r == centroid.r)
			return true;
	}
	return false;
}
float CUDA_K_means(int& colours, unsigned char* pixelTab, int size,bool reduction, int initMethod)
{
	cudaEvent_t startWhole, stopWhole;
	cudaEventCreate(&startWhole);
	cudaEventCreate(&stopWhole);
	float milliseconds = 0;
	cudaEventRecord(startWhole);

	if(reduction)
		cluster(colours, (CudaPixel*)pixelTab, size, initMethod);
	else
		clusterWithoutReduction(colours, (CudaPixel*)pixelTab, size ,initMethod);

	cudaEventRecord(stopWhole);
	cudaEventSynchronize(stopWhole);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, startWhole, stopWhole);
	return milliseconds;
}
void randomInit(thrust::device_vector<CudaPixel>& PixelTab, int k, thrust::device_vector<CudaPixel>& centroids)
{
	std::random_device dev;
	std::mt19937 rng(dev());

	std::uniform_int_distribution<std::mt19937::result_type> randomPixel(0, (int)PixelTab.size() - 1);

	for (int i = 0; i < k; i++)
	{
		int index;
		thrust::host_vector<CudaPixel> dsa(centroids);
		do
		{
			index = randomPixel(rng);
		} while (checkCentroidDuplicate(dsa, PixelTab[index]));
		centroids[i] = (PixelTab[index]);
	}
}
__global__ void calculateNearestCentroids(unsigned char* blueTab, unsigned char* greenTab, unsigned char* redTab, int pixelSize, unsigned char* centroidTab, int centroidSize, int* nearestCentroidTab)
{
	extern __shared__ unsigned char centroids[];
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	if (blockDim.x < centroidSize)
		for (int i = threadIdx.x, j = 0; j <= (centroidSize - 1) / blockDim.x; ++j, i += blockDim.x)
		{
			if (i < centroidSize)
				centroids[i] = centroidTab[i];
		}
	else if (threadIdx.x < centroidSize)
	{
		centroids[threadIdx.x] = centroidTab[threadIdx.x];
	}
	__syncthreads();
	if (index < pixelSize)
	{
		for (int i = index; i < pixelSize; i += stride)
		{
			int nearest = 0;
			int distance = 256 * 256 * 256;
			for (int j = 0; j < centroidSize / 3; j++)
			{
				int newDistance = pixelDistance2(blueTab[i], greenTab[i], redTab[i], centroids[3 * j], centroids[3 * j + 1], centroids[3 * j + 2]);
				if (newDistance < distance)
				{
					distance = newDistance;
					nearest = j;
				}
			}
			nearestCentroidTab[i] = nearest;
		}
	}
}
__global__ void calcualteNewCentroids(unsigned char* centroids,int* blueSum, int* greenSum, int* redSum, int* count,int size)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
	{
		int b = blueSum[index];
		int g = greenSum[index];
		int r = redSum[index];
		int c = count[index];
		centroids[3 * index] = (unsigned char)(b/c);
		centroids[3 * index+1] = (unsigned char)(g / c);
		centroids[3 * index+2] = (unsigned char)(r / c);
	}
}
void inline recalculateCentroids(thrust::device_vector<int>& blueTab, thrust::device_vector<int>& greenTab, thrust::device_vector<int>& redTab,
	thrust::device_vector<CudaPixel>& centroids, thrust::device_vector<int>& uniquePixelCount, thrust::device_vector<int>& nearestCentroidTab,
	thrust::device_vector<int>& blueTabValuesBuffer, thrust::device_vector<int>& greenTabValuesBuffer, thrust::device_vector<int>& redTabValuesBuffer, thrust::device_vector<int>& uniquePixelCountBuffer, thrust::device_vector<int>& newIndexes,
	thrust::device_vector<int>& blueValuesSum, thrust::device_vector<int>& greenValuesSum, thrust::device_vector<int>& redValuesSum, thrust::device_vector<int>& uniquePixelCountSum, thrust::device_vector<int> &centroidNumber)
{
	thrust::sequence(thrust::device, newIndexes.begin(), newIndexes.end(), 0, 1);
	thrust::sort_by_key(thrust::device, nearestCentroidTab.begin(), nearestCentroidTab.end(), newIndexes.begin());
	thrust::gather(thrust::device, newIndexes.begin(), newIndexes.end(), blueTab.begin(), blueTabValuesBuffer.begin());
	thrust::gather(thrust::device, newIndexes.begin(), newIndexes.end(), greenTab.begin(), greenTabValuesBuffer.begin());
	thrust::gather(thrust::device, newIndexes.begin(), newIndexes.end(), redTab.begin(), redTabValuesBuffer.begin());
	thrust::gather(thrust::device, newIndexes.begin(), newIndexes.end(), uniquePixelCount.begin(), uniquePixelCountBuffer.begin());

	thrust::reduce_by_key(thrust::device, nearestCentroidTab.begin(), nearestCentroidTab.end(), blueTabValuesBuffer.begin(), centroidNumber.begin(), blueValuesSum.begin());
	thrust::reduce_by_key(thrust::device, nearestCentroidTab.begin(), nearestCentroidTab.end(), greenTabValuesBuffer.begin(), centroidNumber.begin(), greenValuesSum.begin());
	thrust::reduce_by_key(thrust::device, nearestCentroidTab.begin(), nearestCentroidTab.end(), redTabValuesBuffer.begin(), centroidNumber.begin(), redValuesSum.begin());
	thrust::reduce_by_key(thrust::device, nearestCentroidTab.begin(), nearestCentroidTab.end(), uniquePixelCountBuffer.begin(), centroidNumber.begin(), uniquePixelCountSum.begin());


	//for (int i = 0; i < centroids.size(); ++i)
	//{
	//	centroids[i] = CudaPixel({ (unsigned char)(blueValuesSum[i] / uniquePixelCountSum[i]),(unsigned char)(greenValuesSum[i] / uniquePixelCountSum[i]),(unsigned char)(redValuesSum[i] / uniquePixelCountSum[i]) });
	//}
	calcualteNewCentroids << <(255 + centroids.size()) /256,256>> > ((unsigned char*)thrust::raw_pointer_cast(centroids.data()), thrust::raw_pointer_cast(blueValuesSum.data()), thrust::raw_pointer_cast(greenValuesSum.data()), thrust::raw_pointer_cast(redValuesSum.data()), thrust::raw_pointer_cast(uniquePixelCountSum.data()), centroids.size());

}
void inline recalculateCentroidsNoReduction(thrust::device_vector<int>& blueTab, thrust::device_vector<int>& greenTab, thrust::device_vector<int>& redTab,
	thrust::device_vector<CudaPixel>& centroids, thrust::device_vector<int>& nearestCentroidTab,
	thrust::device_vector<int>& blueTabValuesBuffer, thrust::device_vector<int>& greenTabValuesBuffer, thrust::device_vector<int>& redTabValuesBuffer, thrust::device_vector<int>& newIndexes,
	thrust::device_vector<int>& blueValuesSum, thrust::device_vector<int>& greenValuesSum, thrust::device_vector<int>& redValuesSum, thrust::device_vector<int>& PixelCountSum, thrust::device_vector<int>& centroidNumber)
{
	thrust::sequence(thrust::device, newIndexes.begin(), newIndexes.end(), 0, 1);
	thrust::sort_by_key(thrust::device, nearestCentroidTab.begin(), nearestCentroidTab.end(), newIndexes.begin());
	thrust::gather(thrust::device, newIndexes.begin(), newIndexes.end(), blueTab.begin(), blueTabValuesBuffer.begin());
	thrust::gather(thrust::device, newIndexes.begin(), newIndexes.end(), greenTab.begin(), greenTabValuesBuffer.begin());
	thrust::gather(thrust::device, newIndexes.begin(), newIndexes.end(), redTab.begin(), redTabValuesBuffer.begin());

	thrust::reduce_by_key(thrust::device, nearestCentroidTab.begin(), nearestCentroidTab.end(), blueTabValuesBuffer.begin(), centroidNumber.begin(), blueValuesSum.begin());
	thrust::reduce_by_key(thrust::device, nearestCentroidTab.begin(), nearestCentroidTab.end(), greenTabValuesBuffer.begin(), centroidNumber.begin(), greenValuesSum.begin());
	thrust::reduce_by_key(thrust::device, nearestCentroidTab.begin(), nearestCentroidTab.end(), redTabValuesBuffer.begin(), centroidNumber.begin(), redValuesSum.begin());
	thrust::reduce_by_key(thrust::device, nearestCentroidTab.begin(), nearestCentroidTab.end(), thrust::constant_iterator<int>(1), centroidNumber.begin(), PixelCountSum.begin());

	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	////time start
	//cudaEventRecord(start);

	calcualteNewCentroids << <(255 + centroids.size()) / 256, 256 >> > ((unsigned char*)thrust::raw_pointer_cast(centroids.data()), thrust::raw_pointer_cast(blueValuesSum.data()), thrust::raw_pointer_cast(greenValuesSum.data()), thrust::raw_pointer_cast(redValuesSum.data()), thrust::raw_pointer_cast(PixelCountSum.data()), centroids.size());

	////time stop
	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	//float milliseconds = 0;
	//cudaEventElapsedTime(&milliseconds, start, stop);
	//printf("recalculateCentroids %f\n", milliseconds);
}
bool checkCentroidChange(thrust::host_vector<CudaPixel>& centroids, thrust::host_vector<CudaPixel>& oldCentroids)
{
	for (int i = 0; i < centroids.size(); i++)
	{
		int distanceBlue = centroids[i].b - oldCentroids[i].b;
		int distanceGreen = centroids[i].g - oldCentroids[i].g;
		int distanceRed = centroids[i].r - oldCentroids[i].r;
		int shift = 1;
		if ((distanceBlue || distanceGreen || distanceRed))
		{
			if (distanceBlue * distanceBlue > shift || distanceGreen * distanceGreen > shift || distanceRed * distanceRed > shift)
				return true;
		}
	}
	return false;
}
__global__ void calculateNearestReducedKernel(unsigned char* blue, unsigned char* green, unsigned char* red, int size, unsigned char* centroidData, unsigned long long int* shortestDistance,int* pixelCount)
{
	extern __shared__ unsigned char centroid[];
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	if (threadIdx.x < 3)
	{
		centroid[threadIdx.x] = centroidData[threadIdx.x];
	}
	__syncthreads();
	if (index < size)
	{
		for (int i = index; i < size; i += stride)
		{
			unsigned long long int newDistance = pixelDistance2(blue[i], green[i], red[i], centroid[0], centroid[1], centroid[2]) * pixelCount[i];
			if (newDistance < shortestDistance[i])
				shortestDistance[i] = newDistance;
		}
	}
}
__global__ void calculateNearestKernel(unsigned char* blue, unsigned char* green, unsigned char* red, int size,unsigned char* centroidData,unsigned int * shortestDistance)
{
	extern __shared__ unsigned char centroid[];
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	if (threadIdx.x < 3)
	{
		centroid[threadIdx.x] = centroidData[threadIdx.x];
	}
	__syncthreads();
	if (index < size)
	{
		for (int i = index; i < size; i += stride)
		{
			int newDistance = pixelDistance2(blue[i], green[i], red[i], centroid[0], centroid[1], centroid[2]);
			if (newDistance < shortestDistance[i])
				shortestDistance[i] = newDistance;
		}
	}
}
void kppInit(thrust::device_vector<unsigned char>& blueTab, thrust::device_vector<unsigned char>& greenTab, thrust::device_vector<unsigned char>& redTab, thrust::device_vector<CudaPixel>& centroids)
{
	int threadsPerBlock = 256;
	int blocksPerSM = 32;
	int numberOfSM = 13;
	thrust::host_vector<CudaPixel> newCentroids;
	thrust::device_vector<unsigned int> shortestDistances(blueTab.size());
	thrust::device_vector<unsigned long long> cumulativeDistances(blueTab.size());
	thrust::sequence(shortestDistances.begin(), shortestDistances.end(), 256 * 256 * 256, 0);
	unsigned int currentMax;

	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_int_distribution<std::mt19937::result_type> randomPixel(0, (int)greenTab.size() - 1);
	int index = randomPixel(rng);
	centroids[0] = (CudaPixel({ blueTab [index],greenTab[index] ,redTab[index] }));

	for (int i = 1; i < centroids.size(); ++i)
	{

		calculateNearestKernel << <numberOfSM * blocksPerSM, threadsPerBlock, 3 * sizeof(unsigned char) >> > (thrust::raw_pointer_cast(blueTab.data()), thrust::raw_pointer_cast(greenTab.data()), thrust::raw_pointer_cast(redTab.data()),
			blueTab.size(), (unsigned char*)(thrust::raw_pointer_cast(centroids.data())+i-1), thrust::raw_pointer_cast(shortestDistances.data()));

		thrust::inclusive_scan(thrust::device, shortestDistances.begin(), shortestDistances.end(), cumulativeDistances.begin());
		currentMax = cumulativeDistances[cumulativeDistances.size() - 1];

		std::uniform_int_distribution<unsigned long long> randomVal(0, currentMax);
		unsigned long long int random = randomVal(rng);
		auto it = thrust::lower_bound(thrust::device,cumulativeDistances.begin(), cumulativeDistances.end(),random);
		int newIndex = it - cumulativeDistances.begin();

		centroids[i] = (CudaPixel({ blueTab[newIndex],greenTab[newIndex] ,redTab[newIndex] }));
	}

}
void kppInitReduced(thrust::device_vector<unsigned char>& blueTab, thrust::device_vector<unsigned char>& greenTab, thrust::device_vector<unsigned char>& redTab, thrust::device_vector<CudaPixel>& centroids, thrust::device_vector<int>& uniquePixelCount)
{
	int threadsPerBlock = 256;
	int blocksPerSM = 32;
	int numberOfSM = 13;
	thrust::device_vector<unsigned long long int> shortestDistances(blueTab.size());
	thrust::device_vector<unsigned long long int> cumulativeDistances(blueTab.size());
	thrust::sequence(shortestDistances.begin(), shortestDistances.end(),(unsigned long long) LLONG_MAX, (unsigned long long)0);
	unsigned int currentMax;

	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_int_distribution<std::mt19937::result_type> randomPixel(0, (int)greenTab.size() - 1);
	int index = randomPixel(rng);
	centroids[0] = (CudaPixel({ blueTab[index],greenTab[index] ,redTab[index] }));

	for (int i = 1; i < centroids.size(); ++i)
	{

		calculateNearestReducedKernel << <numberOfSM * blocksPerSM, threadsPerBlock, 3 * sizeof(unsigned char) >> > (thrust::raw_pointer_cast(blueTab.data()), thrust::raw_pointer_cast(greenTab.data()), thrust::raw_pointer_cast(redTab.data()),
			blueTab.size(), (unsigned char*)(thrust::raw_pointer_cast(centroids.data()) + i - 1), thrust::raw_pointer_cast(shortestDistances.data()), thrust::raw_pointer_cast(uniquePixelCount.data()));

		thrust::inclusive_scan(thrust::device, shortestDistances.begin(), shortestDistances.end(), cumulativeDistances.begin());
		currentMax = cumulativeDistances[cumulativeDistances.size() - 1];

		std::uniform_int_distribution<unsigned long long> randomVal(0, currentMax);
		unsigned long long int random = randomVal(rng);
		auto it = thrust::lower_bound(thrust::device, cumulativeDistances.begin(), cumulativeDistances.end(), random);
		int newIndex = it - cumulativeDistances.begin();

		centroids[i] = (CudaPixel({ blueTab[newIndex],greenTab[newIndex] ,redTab[newIndex] }));
	}
}
void lloyd(thrust::device_vector<unsigned char>& blueTab, thrust::device_vector<unsigned char>& greenTab, thrust::device_vector<unsigned char>& redTab, thrust::device_vector<CudaPixel>& centroids, thrust::device_vector<int>& uniquePixelCount, thrust::device_vector<int>& nearestCentroidTab, int& iterations)
{
	int threadsPerBlock = 256;
	int blocksPerSM = 32;
	int numberOfSM = 13;
	float err = 10;
	float maxErr = 5;
	thrust::host_vector<CudaPixel> oldCentroids;
	thrust::host_vector<CudaPixel> hostCentroids(centroids);

	thrust::device_vector<int> blueTabValues(blueTab);
	thrust::device_vector<int> greenTabValues(greenTab);
	thrust::device_vector<int> redTabValues(redTab);

	thrust::device_vector<int> blueTabValuesBuffer(blueTabValues.size());
	thrust::device_vector<int> greenTabValuesBuffer(greenTabValues.size());
	thrust::device_vector<int> redTabValuesBuffer(redTabValues.size());
	thrust::device_vector<int> uniquePixelCountBuffer(uniquePixelCount.size());
	thrust::device_vector<int> newIndexes(uniquePixelCount.size());

	thrust::device_vector<int> blueValuesSum(centroids.size());
	thrust::device_vector<int> greenValuesSum(centroids.size());
	thrust::device_vector<int> redValuesSum(centroids.size());
	thrust::device_vector<int> uniquePixelCountSum(centroids.size());
	thrust::device_vector<int> centroidNumber(uniquePixelCountSum.size());
	reducedPixelsValueMultiply << <numberOfSM * blocksPerSM, threadsPerBlock >> > ((unsigned char*)thrust::raw_pointer_cast(blueTab.data()), (unsigned char*)thrust::raw_pointer_cast(greenTab.data()), (unsigned char*)thrust::raw_pointer_cast(redTab.data()), uniquePixelCount.size(), (int*)thrust::raw_pointer_cast(uniquePixelCount.data()), (int*)thrust::raw_pointer_cast(blueTabValues.data()), (int*)thrust::raw_pointer_cast(greenTabValues.data()), (int*)thrust::raw_pointer_cast(redTabValues.data()));
	do
	{
		thrust::device_vector<CudaPixel>deviceCentroids(centroids);
		calculateNearestCentroids << <numberOfSM * blocksPerSM, threadsPerBlock, centroids.size() * 3 * sizeof(unsigned char) >> > (thrust::raw_pointer_cast(blueTab.data()), thrust::raw_pointer_cast(greenTab.data()), thrust::raw_pointer_cast(redTab.data()), blueTab.size(), (unsigned char*)thrust::raw_pointer_cast(deviceCentroids.data()), centroids.size() * 3, thrust::raw_pointer_cast(nearestCentroidTab.data()));


		oldCentroids.swap(hostCentroids);
		recalculateCentroids(blueTabValues, greenTabValues, redTabValues, centroids, uniquePixelCount, nearestCentroidTab, blueTabValuesBuffer, greenTabValuesBuffer, redTabValuesBuffer, uniquePixelCountBuffer, newIndexes, blueValuesSum, greenValuesSum, redValuesSum, uniquePixelCountSum, centroidNumber);
		hostCentroids = centroids;


	} while (iterations-- && err > maxErr && checkCentroidChange(hostCentroids, oldCentroids));
}
void noReductionLloyd(thrust::device_vector<unsigned char>& blueTab, thrust::device_vector<unsigned char>& greenTab, thrust::device_vector<unsigned char>& redTab, thrust::device_vector<CudaPixel>& centroids, thrust::device_vector<int>& nearestCentroidTab, int& iterations)
{
	int threadsPerBlock = 256;
	int blocksPerSM = 32;
	int numberOfSM = 13;
	float err = 10;
	float maxErr = 5;
	thrust::host_vector<CudaPixel> oldCentroids;
	thrust::host_vector<CudaPixel> hostCentroids(centroids);

	thrust::device_vector<int> blueTabValues(blueTab);
	thrust::device_vector<int> greenTabValues(greenTab);
	thrust::device_vector<int> redTabValues(redTab);

	thrust::device_vector<int> blueTabValuesBuffer(blueTabValues.size());
	thrust::device_vector<int> greenTabValuesBuffer(greenTabValues.size());
	thrust::device_vector<int> redTabValuesBuffer(redTabValues.size());
	thrust::device_vector<int> newIndexes(redTabValues.size());

	thrust::device_vector<int> blueValuesSum(centroids.size());
	thrust::device_vector<int> greenValuesSum(centroids.size());
	thrust::device_vector<int> redValuesSum(centroids.size());
	thrust::device_vector<int> PixelCountSum(centroids.size());
	thrust::device_vector<int> centroidNumber(PixelCountSum.size());
	do
	{
		thrust::device_vector<CudaPixel>deviceCentroids(centroids);
		calculateNearestCentroids << <numberOfSM * blocksPerSM, threadsPerBlock, centroids.size() * 3 * sizeof(unsigned char) >> > (thrust::raw_pointer_cast(blueTab.data()), thrust::raw_pointer_cast(greenTab.data()), thrust::raw_pointer_cast(redTab.data()), blueTab.size(), (unsigned char*)thrust::raw_pointer_cast(deviceCentroids.data()), centroids.size() * 3, thrust::raw_pointer_cast(nearestCentroidTab.data()));


		oldCentroids.swap(hostCentroids);
		recalculateCentroidsNoReduction(blueTabValues, greenTabValues, redTabValues, centroids, nearestCentroidTab, blueTabValuesBuffer, greenTabValuesBuffer, redTabValuesBuffer, newIndexes, blueValuesSum, greenValuesSum, redValuesSum, PixelCountSum, centroidNumber);
		hostCentroids = centroids;


	} while (iterations-- && err > maxErr && checkCentroidChange(hostCentroids, oldCentroids));
}

__host__ void cluster(int& colours, CudaPixel* pixelTab, int size, int initMethod)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;

	int threadsPerBlock = 256;
	int blocksPerSM = 32;
	int numberOfSM = 13;

	int iterations = 200;
	cudaEventRecord(start);
	unsigned int newSize = size / 3;
	thrust::device_vector<CudaPixel> originalPixelTab((CudaPixel*)pixelTab, (CudaPixel*)pixelTab + newSize);
	cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaEventRecord(start);
	thrust::device_vector<CudaPixel> uniquePixelTab(originalPixelTab.size());
	thrust::device_vector<int> uniquePixelCount(originalPixelTab.size());
	thrust::device_vector<CudaPixel> uniquePixelTabCopy(originalPixelTab);

	thrust::sort(thrust::device, uniquePixelTabCopy.begin(), uniquePixelTabCopy.end(), pixelLess());
	auto new_end = thrust::reduce_by_key(uniquePixelTabCopy.begin(), uniquePixelTabCopy.end(), thrust::constant_iterator<int>(1), uniquePixelTab.begin(), uniquePixelCount.begin(), pixelEqual());
	uniquePixelTab.resize(new_end.first - uniquePixelTab.begin());
	uniquePixelCount.resize(new_end.second - uniquePixelCount.begin());
	cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	//SOA
	cudaEventRecord(start);
	thrust::device_vector<unsigned char> blueTab(uniquePixelTab.size());
	thrust::device_vector<unsigned char> greenTab(uniquePixelTab.size());
	thrust::device_vector<unsigned char> redTab(uniquePixelTab.size());

	KMeansConvertToSOA << <numberOfSM * blocksPerSM, threadsPerBlock >> > (
		(unsigned char*)thrust::raw_pointer_cast(uniquePixelTab.data()),
		thrust::raw_pointer_cast(blueTab.data()),
		thrust::raw_pointer_cast(greenTab.data()),
		thrust::raw_pointer_cast(redTab.data()),
		uniquePixelTab.size());

	cudaEventRecord(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	//init
	cudaEventRecord(start);
	thrust::device_vector<CudaPixel> centroids(colours);
	if (initMethod == 1)
		randomInit(uniquePixelTab, colours, centroids);
	else if (initMethod == 2)
		kppInitReduced(blueTab, greenTab, redTab, centroids, uniquePixelCount);
	thrust::device_vector<int> nearestCentroid(uniquePixelTab.size());
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	//Loop
	cudaEventRecord(start);
	lloyd(blueTab, greenTab, redTab, centroids, uniquePixelCount, nearestCentroid, iterations);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaEventRecord(start);
	//Assign new values
	thrust::device_vector<CudaPixel> deviceCentroids(centroids);
	assignNewValues << <numberOfSM * blocksPerSM, threadsPerBlock, deviceCentroids.size() * 3 * sizeof(unsigned char) >> > (
		(unsigned char*)thrust::raw_pointer_cast(originalPixelTab.data()), originalPixelTab.size(),
		(unsigned char*)thrust::raw_pointer_cast(deviceCentroids.data()),
		centroids.size() * 3);

	cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventRecord(start);
	thrust::copy(originalPixelTab.begin(), originalPixelTab.end(), pixelTab);
	cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
}
__host__ void clusterWithoutReduction(int& colours, CudaPixel* pixelTab, int size, int initMethod)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;

	int threadsPerBlock = 256;
	int blocksPerSM = 32;
	int numberOfSM = 13;

	int iterations = 200;

	cudaEventRecord(start);
	unsigned int newSize = size / 3;
	thrust::device_vector<CudaPixel> originalPixelTab((CudaPixel*)pixelTab, (CudaPixel*)pixelTab + newSize);
	cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaEventRecord(start);
	//SOA
	thrust::device_vector<unsigned char> blueTab(originalPixelTab.size());
	thrust::device_vector<unsigned char> greenTab(originalPixelTab.size());
	thrust::device_vector<unsigned char> redTab(originalPixelTab.size());

	KMeansConvertToSOA << <numberOfSM * blocksPerSM, threadsPerBlock >> > (
		(unsigned char*)thrust::raw_pointer_cast(originalPixelTab.data()),
		thrust::raw_pointer_cast(blueTab.data()),
		thrust::raw_pointer_cast(greenTab.data()),
		thrust::raw_pointer_cast(redTab.data()),
		originalPixelTab.size());
	cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaEventRecord(start);
	//init
	thrust::device_vector<CudaPixel> centroids(colours);
	if (initMethod == 1)
		randomInit(originalPixelTab, colours, centroids);
	else if (initMethod == 2)
		kppInit(blueTab, greenTab, redTab,centroids);
	cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	thrust::device_vector<int> nearestCentroid(originalPixelTab.size());


	//Loop
	cudaEventRecord(start);
	noReductionLloyd(blueTab, greenTab, redTab, centroids, nearestCentroid, iterations);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaEventRecord(start);
	//Assign new values
	thrust::device_vector<CudaPixel> deviceCentroids(centroids);
	assignNewValues << <numberOfSM * blocksPerSM, threadsPerBlock, deviceCentroids.size() * 3 * sizeof(unsigned char) >> > (
		(unsigned char*)thrust::raw_pointer_cast(originalPixelTab.data()), originalPixelTab.size(),
		(unsigned char*)thrust::raw_pointer_cast(deviceCentroids.data()),
		centroids.size() * 3);
	cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventRecord(start);
	thrust::copy(originalPixelTab.begin(), originalPixelTab.end(), pixelTab);
	cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
}
