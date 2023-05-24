#pragma once

struct programParams {
	bool cuda = false;
	bool cpu = false;
	bool bMedianCut = false;
	bool bKMeans = false;
	int numColors = 64;
	std::string image = "cyberpunk.bmp";
	int iterations = 200;
	bool reduce = false;
	int method = 0;
	float oversampling = 2.0;
};