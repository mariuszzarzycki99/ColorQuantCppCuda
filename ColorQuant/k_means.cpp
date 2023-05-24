#include "k_means.hpp"

#include <cmath>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <random>
#include <limits>
#include <chrono>
KMeans::~KMeans()
{
	delete uniquePixelCount;
	delete uniquePixelTab;
	delete pixelsClusterTab;
}
KMeans::KMeans(std::vector<Pixel>* pPixelTab)
{
	pixelTab = pPixelTab;
}

void KMeans::ReduceData()
{
	isReduced = true;
	std::vector<Pixel> copyPixelTab(*pixelTab);
	std::sort((&copyPixelTab)->begin(), (&copyPixelTab)->end(), [](const Pixel& first, const Pixel& second)
		{
			if (first.blue < second.blue) return true;
			if (first.blue > second.blue) return false;

			if (first.green < second.green) return true;
			if (first.green > second.green) return false;

			if (first.red < second.red) return true;
			if (first.red > second.red) return false;

			return false; });
	uniquePixelCount = new std::vector<int>();
	uniquePixelTab = new std::vector<Pixel>();
	for (auto it = (&copyPixelTab)->begin(); it != (&copyPixelTab)->end();)
	{
		Pixel tmp = (*it);
		int count = 0;

		for (; it != (&copyPixelTab)->end() && tmp.blue == it->blue && tmp.green == it->green && tmp.red == it->red; ++it)
			count++;

		uniquePixelTab->push_back(tmp);
		uniquePixelCount->push_back(count);
		/*printf("B:%d G:%d R:%d count:%d\n", tmp.blue,tmp.green,tmp.red,count);*/
	}

	printf("UNIQUE: %llu \n", uniquePixelCount->size());

	return;
}
void KMeans::ClusterData(int k, int method, float err, int maxIter, bool reduceData, float oversampling)
{
	auto startWhole = std::chrono::high_resolution_clock::now();
	if (reduceData)
	{
		auto start = std::chrono::high_resolution_clock::now();
		ReduceData();
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
		//printf("Reduction time: %lld ms\n", duration.count());
	}

	auto start = std::chrono::high_resolution_clock::now();
	/* initialization of centroids */
	if (method == 1) KMeansInit(k);
	else if (method == 2) KMeansPlusPlusInit(k);
	else if (method == 3) KMeansParallelInit(k, oversampling);
	
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

	start = std::chrono::high_resolution_clock::now();
	KMeansLoop(k, err, maxIter);
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	//printf("Main loop time: %lld ms\n", duration.count());

	start = std::chrono::high_resolution_clock::now();
	AssignNewPixelValues();
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	//printf("Assing new colours time: %lld ms\n", duration.count());

	auto stopWhole = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stopWhole - startWhole);
	//printf("WHOLE TIME: %lld ms\n", duration.count());
	return;
}
int PixelDistance(const Pixel& a, const Centroid& b)
{
	int blueDist = a.blue - b.blue;
	int greenDist = a.green - b.green;
	int redDist = a.red - b.red;
	return blueDist * blueDist + greenDist * greenDist + redDist * redDist;
}
int KMeans::NearestCentroid(const Pixel& a)
{
	int nearest = 0;
	int minDistance = INT_MAX;

	for (int i = 0; i < centroids.size(); ++i)
	{
		int distance = PixelDistance(a, centroids[i]);
		if (distance < minDistance)
		{
			minDistance = distance;
			nearest = i;
		}
	}
	return nearest;
}
bool KMeans::CentroidChanged()
{
	for (int i = 0; i < centroids.size(); i++)
	{
		int distanceBlue = centroids[i].blue - centroids[i].oldBlue;
		int distanceGreen = centroids[i].green - centroids[i].oldGreen;
		int distanceRed = centroids[i].red - centroids[i].oldRed;
		int shift = 1;
		if (centroids[i].numberOfPoints && (distanceBlue || distanceGreen || distanceRed))
		{
			if (distanceBlue * distanceBlue > shift || distanceGreen * distanceGreen > shift || distanceRed * distanceRed > shift)
				return true;
		}
	}
	return false;
}
void KMeans::RecalculateCentroids()
{
	for (int i = 0; i < centroids.size(); i++)
	{
		centroids[i].oldBlue = centroids[i].blue;
		centroids[i].oldGreen = centroids[i].green;
		centroids[i].oldRed = centroids[i].red;
		centroids[i].blue = 0;
		centroids[i].green = 0;
		centroids[i].red = 0;
		centroids[i].numberOfPoints = 0;
	}
	for (int i = 0; i < pixelsClusterTab->size(); i++)
	{
		if (isReduced)
		{
			int centroidIndex = (*pixelsClusterTab)[i];
			centroids[centroidIndex].blue += ((*uniquePixelTab)[i].blue * (*uniquePixelCount)[i]);
			centroids[centroidIndex].green += ((*uniquePixelTab)[i].green * (*uniquePixelCount)[i]);
			centroids[centroidIndex].red += ((*uniquePixelTab)[i].red * (*uniquePixelCount)[i]);
			centroids[centroidIndex].numberOfPoints += (*uniquePixelCount)[i];
		}
		else
		{
			int centroidIndex = (*pixelsClusterTab)[i];
			centroids[centroidIndex].blue += (*pixelTab)[i].blue;
			centroids[centroidIndex].green += (*pixelTab)[i].green;
			centroids[centroidIndex].red += (*pixelTab)[i].red;
			centroids[centroidIndex].numberOfPoints++;
		}
	}
	for (int i = 0; i < centroids.size(); i++)
	{
		if (centroids[i].numberOfPoints != 0)
		{
			centroids[i].blue /= centroids[i].numberOfPoints;
			centroids[i].green /= centroids[i].numberOfPoints;
			centroids[i].red /= centroids[i].numberOfPoints;
		}
		else
		{
			centroids[i].blue = centroids[i].oldBlue;
			centroids[i].green = centroids[i].oldGreen;
			centroids[i].red = centroids[i].oldRed;
		}
	}
}
void KMeans::KMeansLoop(int k, float err, int maxIter)
{
	float current_error = std::numeric_limits<float>::max();

	std::vector<Pixel>* pixelTabPtr = isReduced ? uniquePixelTab : pixelTab;
	pixelsClusterTab = new std::vector<int>(pixelTabPtr->size());

	float allTimes = 0;
	int iters = 0;
	//printf("Init time: %lld ms\n", duration.count());
	do
	{
		auto start = std::chrono::high_resolution_clock::now();

		for (int i = 0; i < pixelTabPtr->size(); i++)
			(*pixelsClusterTab)[i] = NearestCentroid((*pixelTabPtr)[i]);


		RecalculateCentroids();
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
		allTimes += duration.count();
		iters++;
		//printf("Loop time: %lld ms\n", duration.count());
	} while (((current_error > err) && (maxIter-- != 0) && CentroidChanged()));
	//printf("Loop time: %lld ms\n", duration.count());
}
bool KMeans::CheckCentroidDuplicate(const Pixel& candidate)
{
	for (int i = 0; i < centroids.size(); i++)
	{
		if (candidate.blue == centroids[i].blue && candidate.green == centroids[i].green && candidate.red == centroids[i].red)
			return true;
	}
	return false;
}
void KMeans::KMeansInit(int k)
{
	std::vector<Pixel>* pixelTabPtr = isReduced ? uniquePixelTab : pixelTab;

	std::random_device dev;
	std::mt19937 rng(dev());

	std::uniform_int_distribution<std::mt19937::result_type> randomPixel(0, (int)pixelTabPtr->size() - 1);

	for (int i = 0; i < k; i++)
	{
		int index;
		do
		{
			index = randomPixel(rng);
		} while (CheckCentroidDuplicate((*pixelTabPtr)[index]));

		centroids.emplace_back((*pixelTabPtr)[index].blue, (*pixelTabPtr)[index].green, (*pixelTabPtr)[index].red, 0);
	}

}
int KMeans::bisectionSearch(long long* sumTable, int size, long long searchValue)
{
	int il, ir, i;

	if (size < 1) {
		return 0;
	}

	if (searchValue < sumTable[0]) {
		return 0;
	}
	else if (searchValue > sumTable[size - 1]) {
		return size - 1;
	}

	il = 0;
	ir = size - 1;

	i = (il + ir) / 2;
	while (i != il) {
		if (sumTable[i] <= searchValue) {
			il = i;
		}
		else {
			ir = i;
		}
		i = (il + ir) / 2;
	}

	if (sumTable[i] <= searchValue)
		i = ir;
	return i;
}
void KMeans::KMeansPlusPlusInit(int k)
{
	std::vector<Pixel>* pixelTabPtr = isReduced ? uniquePixelTab : pixelTab;
	std::random_device dev;
	std::mt19937 rng(dev());

	std::uniform_int_distribution<std::mt19937::result_type> randomPixel(0, (int)pixelTabPtr->size() - 1);

	/* first centroid */
	int index = randomPixel(rng);
	centroids.emplace_back((*pixelTabPtr)[index].blue, (*pixelTabPtr)[index].green, (*pixelTabPtr)[index].red, 0);

	std::unique_ptr<long long> shortestDistance = std::make_unique<long long>(pixelTabPtr->size());
	std::unique_ptr<long long> cumulativeDistances = std::make_unique<long long>(pixelTabPtr->size());

	for (int i = 0; i < pixelTabPtr->size(); ++i)
		shortestDistance.get()[i] = LONG_MAX;

	for (int centr = 1; centr < k; centr++)
	{

		/* For each point find its closest distance to any of
		the previous cluster centers */
		for (int i = 0; i < pixelTabPtr->size(); i++)
		{
			int newDistance = PixelDistance((*pixelTabPtr)[i], centroids[centr - 1]);
			if (newDistance < shortestDistance.get()[i])
				shortestDistance.get()[i] = newDistance;
		}

		/* Create an array of the cumulative distances. */
		if (isReduced)
		{
			cumulativeDistances.get()[0] = shortestDistance.get()[0] * (*uniquePixelCount)[0];
			for (int i = 1; i < pixelTabPtr->size(); i++)
				cumulativeDistances.get()[i] = shortestDistance.get()[i] * (*uniquePixelCount)[i] + cumulativeDistances.get()[i - 1];
		}
		cumulativeDistances.get()[0] = shortestDistance.get()[0];
		for (int i = 1; i < pixelTabPtr->size(); i++)
			cumulativeDistances.get()[i] = shortestDistance.get()[i] + cumulativeDistances.get()[i - 1];
		do
		{
			/* Select a point at random. Those with greater distances
			have a greater probability of being selected. */
			std::uniform_int_distribution<unsigned long long> randomSumValue(0, cumulativeDistances.get()[pixelTabPtr->size() - 1]);
			index = bisectionSearch(cumulativeDistances.get(), (int)pixelTabPtr->size(), randomSumValue(rng));
		} while (CheckCentroidDuplicate((*pixelTabPtr)[index]));

		centroids.emplace_back((*pixelTabPtr)[index].blue, (*pixelTabPtr)[index].green, (*pixelTabPtr)[index].red, 0);
	}
}
void KMeans::KMeansParallelInit(int k, float oversampling)
{
	std::vector<Pixel>* pixelTabPtr = isReduced ? uniquePixelTab : pixelTab;
	std::random_device dev;
	std::mt19937 rng(dev());

	std::uniform_int_distribution<std::mt19937::result_type> randomPixel(0, (int)pixelTabPtr->size() - 1);

	std::vector<unsigned int> distances(pixelTabPtr->size(), 0);


	/* first centroid randomly */
	int index = randomPixel(rng);
	centroids.emplace_back((*pixelTabPtr)[index].blue, (*pixelTabPtr)[index].green, (*pixelTabPtr)[index].red, 0);

	unsigned long long sum = 0;
	for (int i = 0; i < pixelTabPtr->size(); i++)
	{
		int distance = PixelDistance((*pixelTabPtr)[i], centroids[0]);
		distances[i] = distance;
		if (isReduced)
			sum += distance * (*uniquePixelCount)[i];
		else
			sum += distance;
	}

	int logSum = (log(sum) + 0.5);

	int centroid_sum = 0;
	for (int i = 0; i <= logSum; i++)
	{
		int centroid_counter = 0;
		int repeats = 0;
		for (int j = 0; j < pixelTabPtr->size(); j++)
		{
			if (isReduced)
			{
				for (int k = 0; k < (*uniquePixelCount)[j]; k++)
				{
					std::uniform_int_distribution<std::mt19937::result_type> randomValue(0, sum);
					unsigned int randomVal = randomValue(rng);
					if (oversampling * (distances[j]) > randomVal)
					{
						if (!CheckCentroidDuplicate((*pixelTabPtr)[j]))
						{
							sum -= distances[j];
							distances[j] = 0;
							centroids.emplace_back((*pixelTabPtr)[j].blue, (*pixelTabPtr)[j].green, (*pixelTabPtr)[j].red, 0);
							centroid_counter++;
						}
						else
						{
							repeats++;
						}
					}
				}
			}
			else
			{
				std::uniform_int_distribution<std::mt19937::result_type> randomValue(0, sum);
				unsigned int randomVal = randomValue(rng);
				if (oversampling * (distances[j]) > randomVal)
				{
					if (!CheckCentroidDuplicate((*pixelTabPtr)[j]))
					{
						sum -= distances[j];
						distances[j] = 0;
						centroids.emplace_back((*pixelTabPtr)[j].blue, (*pixelTabPtr)[j].green, (*pixelTabPtr)[j].red, 0);
						centroid_counter++;
					}
					else
					{
						repeats++;
					}
				}
			}
		}
		centroid_sum += centroid_counter;
		printf("Loop nr %d: %d centroids\n", i, centroid_counter);
		printf("Loop nr %d: %d repeats\n", i, repeats);
	}
	printf("Centroid counter = %d\n", centroid_sum);
	printf("LOGSUM = %d, OVERSAMPLING = %f, LOGSUM * OVERSAMPLING = %f\n", logSum, oversampling, (float)logSum * oversampling);

	std::vector<long long> weights(centroids.size(), 0);
	std::vector<long long> cumulativeWeights(centroids.size(), 0);

	printf("Reduction1\n");
	for (int i = 0; i < pixelTabPtr->size(); i++)
	{
		if (isReduced)
			weights[NearestCentroid((*pixelTabPtr)[i])] += (*uniquePixelCount)[i];
		else
			weights[NearestCentroid((*pixelTabPtr)[i])]++;
	}
	printf("Reduction2\n");
	cumulativeWeights[0] = weights[0];
	for (int i = 1; i < centroids.size(); i++)
	{
		cumulativeWeights[i] = cumulativeWeights[i - 1] + weights[i];
	}
	printf("Reduction3\n");

	std::vector<Centroid> toReduceCentroids(centroids);
	centroids.clear();

	for (int i = 0; i < k; i++)
	{
		std::uniform_int_distribution<long long> randomWeightsSumValue(0, cumulativeWeights[toReduceCentroids.size() - 1]);
		index = bisectionSearch(cumulativeWeights.data(), (int)toReduceCentroids.size(), randomWeightsSumValue(rng));

		for (int i = index; i < cumulativeWeights.size(); i++)
			cumulativeWeights[i] -= weights[index];

		cumulativeWeights.erase(cumulativeWeights.begin() + index);
		weights.erase(weights.begin() + index);

		centroids.emplace_back(toReduceCentroids[index].blue, toReduceCentroids[index].green, toReduceCentroids[index].red, 0);
		toReduceCentroids.erase(toReduceCentroids.begin() + index);
	}
	printf("Reduction4\n");
}
void KMeans::AssignNewPixelValues()
{
	if (isReduced)
	{
		int* tmpPixelsClusterTab = new int[pixelTab->size()];

		for (int i = 0; i < pixelTab->size(); i++)
			tmpPixelsClusterTab[i] = NearestCentroid((*pixelTab)[i]);

		for (int i = 0; i < pixelTab->size(); i++)
		{
			(*pixelTab)[i].blue = centroids[tmpPixelsClusterTab[i]].blue;
			(*pixelTab)[i].green = centroids[tmpPixelsClusterTab[i]].green;
			(*pixelTab)[i].red = centroids[tmpPixelsClusterTab[i]].red;
		}
	}
	else
	{
		for (int i = 0; i < pixelTab->size(); i++)
		{
			(*pixelTab)[i].blue = centroids[(*pixelsClusterTab)[i]].blue;
			(*pixelTab)[i].green = centroids[(*pixelsClusterTab)[i]].green;
			(*pixelTab)[i].red = centroids[(*pixelsClusterTab)[i]].red;
		}
	}
	int sum = 0;
	for (int i = 0; i < centroids.size(); i++)
	{
		sum += centroids[i].numberOfPoints;
		//printf("Color %d: R:%d G:%d B:%d Size:%d\n", i, centroids[i].red, centroids[i].green, centroids[i].blue, centroids[i].numberOfPoints);
	}
}
