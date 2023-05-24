#ifndef BITMAP_HPP
#define BITMAP_HPP

#include <inttypes.h>
#include <vector>
#include <string>
#include <memory>

#pragma pack(push,1)

typedef struct BMPFileHeader
{
	short bfType;
	int bfSize;
	short bfReserved1;
	short bfReserved2;
	int bfOffBits;
}BMPFileHeader;

typedef struct BMPPictureHeader
{
	unsigned int biSize;
	int biWidth;
	int biHeight;
	unsigned short biPlanes;
	unsigned short biBitCount;
	unsigned int biCompression;
	unsigned int biSizeImage;
	int biXPelsPerMeter;
	int biYPelsPerMeter;
	unsigned int biClrUsed;
	unsigned int biClrImportant;
}BMPPictureHeader;


typedef struct Pixel
{
	uint8_t blue;
	uint8_t green;
	uint8_t red;
}Pixel;

typedef struct PixelSOA
{
	size_t N;
	std::unique_ptr<uint8_t> blue;
	std::unique_ptr<uint8_t> green;
	std::unique_ptr<uint8_t> red;
}PixelSOA;

typedef struct Pixel4
{
	uint8_t blue;
	uint8_t green;
	uint8_t red;
	uint8_t unused;
}Pixel4;
#pragma pack(pop)



class BMPImage
{
private:

	BMPFileHeader fileHeader;
	BMPPictureHeader pictureHeader;

	std::unique_ptr<std::vector<Pixel>> pixelTab;
	PixelSOA PixelSOA;

public:
	~BMPImage();
	
	int ReadBMP(std::string filename);
	void ReadBMPasSOA(std::string filename);
	int SaveBMP(std::string filename);
	size_t getPixelTabSize() { return pixelTab->size(); }
	std::vector<Pixel>* getPixelTab();
	void ArrayToSOA();
	
};

#endif 
