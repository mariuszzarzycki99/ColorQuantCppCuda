#include "median_cut.hpp"
#include <algorithm>
#include <chrono>
bool compareB(const Pixel& i, const Pixel& j) { return (i.blue < j.blue); }
bool compareG(const Pixel& i, const Pixel& j) { return (i.green < j.green); }
bool compareR(const Pixel& i, const Pixel& j) { return (i.red < j.red); }
bool (*compare[3]) (const Pixel& i, const Pixel& j) = { compareB ,compareG,compareR };

int PixelDistance(const Pixel& a, const Pixel& b)
{
	int blueDist = a.blue - b.blue;
	int greenDist = a.green - b.green;
	int redDist = a.red - b.red;
	return blueDist * blueDist + greenDist * greenDist + redDist * redDist;
}

int MedianCut::NearestColor(const Pixel& a)
{
	int nearest = 0;
	int minDistance = INT_MAX;

	for (int i = 0; i < newColours.size(); ++i)
	{
		int distance = PixelDistance(a, newColours[i]);
		if (distance < minDistance)
		{
			minDistance = distance;
			nearest = i;
		}
	}
	return nearest;
}
Pixel MedianCut::MeanColor(const Bucket& bucket)
{
	unsigned int b = 0, g = 0, r = 0;
	for (size_t i = bucket.start; i < bucket.end; i++)
	{
		b += (*pixelTabCopy)[i].blue;
		g += (*pixelTabCopy)[i].green;
		r += (*pixelTabCopy)[i].red;
	}

	Pixel pixel;
	pixel.blue = (uint8_t)(b / (bucket.end - bucket.start + 1));
	pixel.green = (uint8_t)(g / (bucket.end - bucket.start + 1));
	pixel.red = (uint8_t)(r / (bucket.end - bucket.start + 1));

	return pixel;
}
int MedianCut::LongestDimension(const Bucket& bucket)
{
	int b_min = 256, g_min = 256, r_min = 256, b_max = 0, g_max = 0, r_max = 0;

	for (size_t i = bucket.start; i < bucket.end; i++)
	{
		if (b_min > (*pixelTabCopy)[i].blue) b_min = (*pixelTabCopy)[i].blue;
		if (g_min > (*pixelTabCopy)[i].green) g_min = (*pixelTabCopy)[i].green;
		if (r_min > (*pixelTabCopy)[i].red) r_min = (*pixelTabCopy)[i].red;

		if (b_max < (*pixelTabCopy)[i].blue) b_max = (*pixelTabCopy)[i].blue;
		if (g_max < (*pixelTabCopy)[i].green) g_max = (*pixelTabCopy)[i].green;
		if (r_max < (*pixelTabCopy)[i].red) r_max = (*pixelTabCopy)[i].red;
	}
	int b_length = b_max - b_min;
	int g_length = g_max - g_min;
	int r_length = r_max - r_min;

	// czerwony najwiekszy
	if (r_length >= g_length)
		if (r_length >= b_length)
			return 3;

	// zielony najwiekszy
	if (g_length >= b_length)
		return 2;

	//niebieski najwiekszy
	return 1;
}
void MedianCut::ApplyColors()
{
#define ATM
#ifdef AT
	for (int i = 0; i < pixelTab->size(); ++i)
	{
		int index = NearestColor(pixelTab->at(i));
		pixelTab->at(i).blue = newColours.at(index).blue;
		pixelTab->at(i).green = newColours.at(index).green;
		pixelTab->at(i).red = newColours.at(index).red;
	}
#endif
#ifndef AT
	for (int i = 0; i < pixelTab->size(); ++i)
	{
		int index = NearestColor((*pixelTab)[i]);
		((*pixelTab)[i]).blue = newColours[index].blue;
		((*pixelTab)[i]).green = newColours[index].green;
		((*pixelTab)[i]).red = newColours[index].red;
	}
#endif
}
void MedianCut::Quant(int colours)
{
	auto start = std::chrono::high_resolution_clock::now();

	int maxDepth = (int)ceil(log2(colours));
	Bucket bucket = { 0,pixelTabCopy->size()};

	SplitBuckets(0, maxDepth,bucket);

	for (auto& bucket : buckets)
		newColours.push_back(MeanColor(bucket));
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	//printf("Median Cut time: %lld ms\n", duration.count());

	start = std::chrono::high_resolution_clock::now();
	ApplyColors();
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	//printf("Apply Colors time: %lld ms\n", duration.count());
}
void MedianCut::SplitBuckets(int depth, int maxDepth, Bucket bucket)
{
	if (depth == maxDepth)
	{
		buckets.push_back(bucket);
		return;
	}
	int longestDim = LongestDimension(bucket);
	/*printf("PRESORT%d\n", depth);
	for (auto a = pixelTabCopy->begin(); a != pixelTabCopy->end(); ++a)
		printf("B:%d G:%d R:%d|\n", a->blue, a->green, a->red);
	printf("\nSORTUJE %d w %d - %d\n", longestDim,bucket.start, bucket.end );*/
	std::stable_sort(pixelTabCopy->begin()+bucket.start, pixelTabCopy->begin()+bucket.end, compare[longestDim-1]);
	Bucket bucket1, bucket2;
	bucket1 = {bucket.start,bucket.start+(bucket.end-bucket.start)/2};
	bucket2 = { bucket.start + (bucket.end - bucket.start) / 2,bucket.end};
	//printf("Bucket 1: %d-%d\nBucket2: %d-%d\n", bucket1.start,bucket1.end,bucket2.start,bucket2.end);
	SplitBuckets(depth + 1, maxDepth, bucket1);
	SplitBuckets(depth + 1, maxDepth, bucket2);
}