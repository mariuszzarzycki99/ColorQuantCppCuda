#ifndef K_MEANS_HPP
#define K_MEANS_HPP
#include "bitmap.hpp"
typedef struct Centroid
{
	unsigned int blue;
	unsigned int green;
	unsigned int red;
	unsigned int numberOfPoints;
	unsigned int oldBlue = 0;
	unsigned int oldGreen = 0;
	unsigned int oldRed = 0;

	Centroid(unsigned int _blue, unsigned int _green, unsigned int _red, unsigned int _numberOfPoints) :blue(_blue), red(_red), green(_green), numberOfPoints(_numberOfPoints) {}
}Centroid;

class KMeans
{
	std::vector<Pixel>* pixelTab;
	std::vector<Pixel>* uniquePixelTab;
	std::vector<int>* uniquePixelCount;
	std::vector<int>* pixelsClusterTab;

	std::vector<Centroid> centroids;

	bool isReduced = false;

	void KMeansLoop(int k, float err, int maxIter);
	void KMeansInit(int k);
	bool CheckCentroidDuplicate(const Pixel& candidate);
	void KMeansPlusPlusInit(int k);
	void KMeansParallelInit(int k, float oversampling);
	void AssignNewPixelValues();
	void ReduceData();

	int NearestCentroid(const Pixel& a);
	bool CentroidChanged();
	void RecalculateCentroids();
	int bisectionSearch(long long* sumTable, int index, long long searchValue);

public:
	~KMeans();
	KMeans(std::vector<Pixel>* pPixelTab);
	void ClusterData(int k, int method, float err, int maxIter, bool reduceData , float oversampling);
};

#endif