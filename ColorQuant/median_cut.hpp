#ifndef MEDIAN_CUT_HPP
#define MEDIAN_CUT_HPP
#include "bitmap.hpp"
#include <memory>

typedef struct Bucket {
	size_t start, end;
}Bucket;


class MedianCut
{
	std::vector<Pixel>* pixelTab;
	std::unique_ptr< std::vector<Pixel>> pixelTabCopy;
	std::vector<Pixel> newColours;
	std::vector<Bucket> buckets;
	int NearestColor(const Pixel& a);
	Pixel MeanColor(const Bucket& bucket);
	int LongestDimension(const Bucket& bucket);
	void SplitBuckets(int depth, int maxDepth, Bucket bucket);
	void ApplyColors();
public:
	MedianCut(std::vector<Pixel>* _pixelTab) : pixelTab(_pixelTab) { 
		pixelTabCopy = std::make_unique<std::vector<Pixel>>(*pixelTab); 
	}
	void Quant(int colours);
	

};

#endif