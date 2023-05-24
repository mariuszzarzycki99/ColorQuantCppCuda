#include "CUDA_median_cut.h"

#include <stdio.h>
#include <device_launch_parameters.h>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/for_each.h>

#define VERBOSE


float CUDA_medianCut(int& colours, uint8_t* pixelTab, int size)
{
	cudaEvent_t startWhole, stopWhole;
	cudaEventCreate(&startWhole);
	cudaEventCreate(&stopWhole);
	float milliseconds = 0;
	cudaEventRecord(startWhole);

	quantize(colours, pixelTab, size);

	cudaEventRecord(stopWhole);
	cudaEventSynchronize(stopWhole);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, startWhole, stopWhole);
	return milliseconds;
	
}
__host__ void splitBuckets(thrust::host_vector<PixelBucket>& buckets)
{
	thrust::host_vector<PixelBucket> newBuckets;
	for (int j = 0; j < buckets.size(); j++)
	{
		int start = buckets[j].start;
		int end = buckets[j].end;
		int middle = (buckets[j].end - buckets[j].start) / 2;
		newBuckets.push_back(PixelBucket({ buckets[j].start,buckets[j].start + middle }));
		newBuckets.push_back(PixelBucket({ buckets[j].start + middle,buckets[j].end }));
	}
	buckets = newBuckets;
}
__host__ void meanColors(thrust::device_vector<unsigned char>& blueTab, thrust::device_vector<unsigned char>& greenTab, thrust::device_vector<unsigned char>& redTab, thrust::host_vector<PixelBucket>& buckets, thrust::host_vector<unsigned char>& newColors)
{
	for (int i = 0; i < buckets.size(); i++)
	{
		unsigned int newB = (unsigned int)thrust::reduce(blueTab.begin() + buckets[i].start, blueTab.begin() + buckets[i].end, (unsigned int)0);
		unsigned int newG = (unsigned int)thrust::reduce(greenTab.begin() + buckets[i].start, greenTab.begin() + buckets[i].end, (unsigned int)0);
		unsigned int newR = (unsigned int)thrust::reduce(redTab.begin() + buckets[i].start, redTab.begin() + buckets[i].end, (unsigned int)0);
		newB /= (buckets[i].end - buckets[i].start + 1);
		newG /= (buckets[i].end - buckets[i].start + 1);
		newR /= (buckets[i].end - buckets[i].start + 1);
		newColors.push_back((unsigned char)newB);
		newColors.push_back((unsigned char)newG);
		newColors.push_back((unsigned char)newR);
	}
}
__global__ void applyColorsKernel(unsigned int* originalIndex, int size, int newColorsSize, int bucketSize, unsigned char* newColors, unsigned char* originalPixelTab)
{
	extern __shared__ unsigned char colors[];
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	if (blockDim.x < newColorsSize)
		for (int i = threadIdx.x, j = 0; j <= (newColorsSize - 1) / blockDim.x; ++j, i += blockDim.x)
		{
			if (i < newColorsSize)
				colors[i] = newColors[i];
		}
	else if (threadIdx.x < newColorsSize)
	{
		colors[threadIdx.x] = newColors[threadIdx.x];
	}
	__syncthreads();
	if (index < size)
	{
		for (int i = index; i < size; i += stride)
		{
			int color = i / bucketSize;
			originalPixelTab[originalIndex[i] * 3] = colors[3 * color];
			originalPixelTab[originalIndex[i] * 3 + 1] = colors[3 * color + 1];
			originalPixelTab[originalIndex[i] * 3 + 2] = colors[3 * color + 2];
		}
	}
}
__global__ void convertToSOA(unsigned char* original, unsigned char* blueTab, unsigned char* greenTab, unsigned char* redTab, int size)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	if (index < size)
	{
		for (int i = index; i < size / 3; i += stride)
		{
			blueTab[i] = original[3 * i];
			greenTab[i] = original[3 * i + 1];
			redTab[i] = original[3 * i + 2];
		}
	}
}
struct sortBucketsFunctor
{
	thrust::device_vector<unsigned char>& blueTab;
	thrust::device_vector<unsigned char>& greenTab;
	thrust::device_vector<unsigned char>& redTab;
	thrust::device_vector<unsigned char>& blueCopy;
	thrust::device_vector<unsigned char>& redCopy;
	thrust::device_vector<unsigned char>& greenCopy;
	thrust::device_vector<unsigned int>& newPosition;

	__host__ sortBucketsFunctor(
		thrust::device_vector<unsigned char>& _blueTab,
		thrust::device_vector<unsigned char>& _greenTab,
		thrust::device_vector<unsigned char>& _redTab,
		thrust::device_vector<unsigned char>& _blueCopy,
		thrust::device_vector<unsigned char>& _redCopy,
		thrust::device_vector<unsigned char>& _greenCopy,
		thrust::device_vector<unsigned int>& _newPosition): blueTab(_blueTab), greenTab(_greenTab),redTab(_redTab),blueCopy(_blueCopy),greenCopy(_greenCopy),redCopy(_redCopy),newPosition(_newPosition){}
	__host__ void operator()(PixelBucket bucket)
	{
		thrust::pair<thrust::device_vector<unsigned char>::iterator, thrust::device_vector<unsigned char>::iterator> blueMinMax;
		thrust::pair<thrust::device_vector<unsigned char>::iterator, thrust::device_vector<unsigned char>::iterator> greenMinMax;
		thrust::pair<thrust::device_vector<unsigned char>::iterator, thrust::device_vector<unsigned char>::iterator> redMinMax;

		int b_length = 0;
		int g_length = 0;
		int r_length = 0;

		blueMinMax = thrust::minmax_element(blueTab.begin() + bucket.start, blueTab.begin() + bucket.end);
		greenMinMax = thrust::minmax_element(greenTab.begin() + bucket.start, greenTab.begin() + bucket.end);
		redMinMax = thrust::minmax_element(redTab.begin() + bucket.start, redTab.begin() + bucket.end);

		b_length = *(blueMinMax.second) - *(blueMinMax.first);
		g_length = *(greenMinMax.second) - *(greenMinMax.first);
		r_length = *(redMinMax.second) - *(redMinMax.first);

		//First, sort the keys and indices by the keys
		if (r_length >= g_length && r_length >= b_length)
		{
			thrust::sort_by_key(thrust::device, (redCopy.begin() + bucket.start), (redCopy.begin() + bucket.end), newPosition.begin() + bucket.start);
		}
		else if (g_length >= b_length && g_length >= r_length)
		{
			thrust::sort_by_key(thrust::device, (greenCopy.begin() + bucket.start), (greenCopy.begin() + bucket.end), newPosition.begin() + bucket.start);
		}
		else if (b_length >= g_length && b_length >= r_length)
		{
			thrust::sort_by_key(thrust::device, (blueCopy.begin() + bucket.start), (blueCopy.begin() + bucket.end), newPosition.begin() + bucket.start);
		}
	}
};

__host__ void sortArrays(
	thrust::device_vector<unsigned char>& blueTab,
	thrust::device_vector<unsigned char>& greenTab,
	thrust::device_vector<unsigned char>& redTab,
	thrust::device_vector<unsigned int>& originalIndex,
	thrust::host_vector<PixelBucket>& buckets,
	thrust::device_vector<unsigned char>& blueCopy,
	thrust::device_vector<unsigned char>& redCopy,
	thrust::device_vector<unsigned char>& greenCopy,
	thrust::device_vector<unsigned int>& originalIndexCopy,
	thrust::device_vector<unsigned int>& newPosition)
{
	thrust::sequence(thrust::device, newPosition.begin(), newPosition.end(), 0, 1);

	thrust::copy(blueTab.begin(), blueTab.end(), blueCopy.begin());
	thrust::copy(redTab.begin(), redTab.end(), redCopy.begin());
	thrust::copy(greenTab.begin(), greenTab.end(), greenCopy.begin());
	thrust::copy(originalIndex.begin(), originalIndex.end(), originalIndexCopy.begin());

	thrust::pair<thrust::device_vector<unsigned char>::iterator, thrust::device_vector<unsigned char>::iterator> blueMinMax;
	thrust::pair<thrust::device_vector<unsigned char>::iterator, thrust::device_vector<unsigned char>::iterator> greenMinMax;
	thrust::pair<thrust::device_vector<unsigned char>::iterator, thrust::device_vector<unsigned char>::iterator> redMinMax;

	int b_length = 0;
	int g_length = 0;
	int r_length = 0;

	for (int j = 0; j < buckets.size(); j++)
	{

		blueMinMax = thrust::minmax_element(blueTab.begin() + buckets[j].start, blueTab.begin() + buckets[j].end);
		greenMinMax = thrust::minmax_element(greenTab.begin() + buckets[j].start, greenTab.begin() + buckets[j].end);
		redMinMax = thrust::minmax_element(redTab.begin() + buckets[j].start, redTab.begin() + buckets[j].end);


		b_length = *(blueMinMax.second) - *(blueMinMax.first);
		g_length = *(greenMinMax.second) - *(greenMinMax.first);
		r_length = *(redMinMax.second) - *(redMinMax.first);



		//First, sort the keys and indices by the keys
		if (r_length >= g_length && r_length >= b_length)
		{
			thrust::sort_by_key(thrust::device, (redCopy.begin() + buckets[j].start), (redCopy.begin() + buckets[j].end), newPosition.begin() + buckets[j].start);
		}
		else if (g_length >= b_length && g_length >= r_length)
		{
			thrust::sort_by_key(thrust::device, (greenCopy.begin() + buckets[j].start), (greenCopy.begin() + buckets[j].end), newPosition.begin() + buckets[j].start);
		}
		else if (b_length >= g_length && b_length >= r_length)
		{
			thrust::sort_by_key(thrust::device, (blueCopy.begin() + buckets[j].start), (blueCopy.begin() + buckets[j].end), newPosition.begin() + buckets[j].start);
		}
	}

	//thrust::for_each(buckets.begin(), buckets.end(), sortBucketsFunctor(blueTab,greenTab,redTab,blueCopy,redCopy,greenCopy,newPosition));

	thrust::gather(newPosition.begin(), newPosition.end(), redTab.begin(), redCopy.begin());
	redTab.swap(redCopy);
	thrust::gather(newPosition.begin(), newPosition.end(), greenTab.begin(), greenCopy.begin());
	greenTab.swap(greenCopy);
	thrust::gather(newPosition.begin(), newPosition.end(), blueTab.begin(), blueCopy.begin());
	blueTab.swap(blueCopy);
	thrust::gather(newPosition.begin(), newPosition.end(), originalIndexCopy.begin(), originalIndex.begin());
}
__host__ void quantize(int& colours, uint8_t* pixelTab, int size)
{
	int threadsPerBlock = 256;
	int blocksPerSM = 32;
	int numberOfSM = 13;

	thrust::device_vector<unsigned char> originalPixelTab((unsigned char*)pixelTab, (unsigned char*)pixelTab + size);
	unsigned int newSize = size / 3;

	thrust::device_vector<unsigned char> blueTab(newSize);
	thrust::device_vector<unsigned char> greenTab(newSize);
	thrust::device_vector<unsigned char> redTab(newSize);

	thrust::device_vector<unsigned int> originalIndex(newSize);
	thrust::sequence(thrust::device, originalIndex.begin(), originalIndex.end(), 0, 1);

	convertToSOA << < numberOfSM * blocksPerSM, threadsPerBlock >> > (thrust::raw_pointer_cast(originalPixelTab.data()), thrust::raw_pointer_cast(blueTab.data()), thrust::raw_pointer_cast(greenTab.data()), thrust::raw_pointer_cast(redTab.data()), size);

	thrust::host_vector<PixelBucket> buckets;
	thrust::host_vector<unsigned char> newColors;
	buckets.push_back(PixelBucket({ 0,newSize }));

	int maxDepth = (int)ceil(log2(colours));

	thrust::device_vector<unsigned char> blueCopy(newSize);
	thrust::device_vector<unsigned char> redCopy(newSize);
	thrust::device_vector<unsigned char> greenCopy(newSize);
	thrust::device_vector<unsigned int> originalIndexCopy(newSize);
	thrust::device_vector<unsigned int> newPosition(newSize);


	for (int i = 0; i <= maxDepth; i++)
	{
		if (i == maxDepth)
		{
			meanColors(blueTab, greenTab, redTab, buckets, newColors);
			break;
		}
		sortArrays(blueTab, greenTab, redTab, originalIndex, buckets, blueCopy, redCopy, greenCopy, originalIndexCopy, newPosition);
		splitBuckets(buckets);
	}
	thrust::device_vector<unsigned char> deviceNewColors(newColors);

	//assign new colors
	applyColorsKernel << <numberOfSM * blocksPerSM, threadsPerBlock, deviceNewColors.size() * sizeof(unsigned char) >> > (thrust::raw_pointer_cast(originalIndex.data()), newSize, deviceNewColors.size(), buckets[0].end - buckets[0].start + 1, (unsigned char*)thrust::raw_pointer_cast(deviceNewColors.data()), thrust::raw_pointer_cast(originalPixelTab.data()));
	thrust::copy(originalPixelTab.begin(), originalPixelTab.end(), pixelTab);
}
