#include "bitmap.hpp"



//#define PRINT_DEBUG_INFO
BMPImage::~BMPImage()
{
}
std::vector<Pixel>* BMPImage::getPixelTab()
{
	return pixelTab.get();
}
int BMPImage::ReadBMP(std::string filename)
{
	errno_t err;
	FILE* file;
	if ((err = fopen_s(&file, filename.c_str(), "rb")) != 0)
	{
		char buf[95] = { '\0' };
		strerror_s(buf, sizeof buf, err);
		fprintf_s(stderr, "cannot open file '%s': %s\n",
			filename.c_str(), buf);
	}
	else
	{
		fread(&fileHeader, sizeof(BMPFileHeader), 1, file);
		fseek(file, 14, SEEK_SET);
		fread(&pictureHeader, sizeof(BMPPictureHeader), 1, file);

		uint8_t padding = pictureHeader.biWidth & 0x03;

		pixelTab = std::unique_ptr<std::vector<Pixel>>(new std::vector<Pixel>((long long)pictureHeader.biWidth * (long long)pictureHeader.biHeight));

		fseek(file, fileHeader.bfOffBits, SEEK_SET);

		for (int i = 0; i < pictureHeader.biHeight; i++)
		{
			fread(pixelTab->data() + i * pictureHeader.biWidth, sizeof(Pixel), pictureHeader.biWidth, file);
			fseek(file, padding, SEEK_CUR);
		}
		fclose(file);
	}
	return 0;
}
int BMPImage::SaveBMP(std::string filename)
{
	errno_t err;
	FILE* file;
	if ((err = fopen_s(&file, filename.c_str(), "wb")) != 0)
	{
		char buf[95] = { '\0' };
		strerror_s(buf, sizeof buf, err);
		fprintf_s(stderr, "cannot open file '%s': %s\n",
			filename.c_str(), buf);
	}
	else
	{
		fwrite(&fileHeader, sizeof(BMPFileHeader), 1, file);
		fwrite(&pictureHeader, sizeof(BMPPictureHeader), 1, file);

		uint8_t padding = pictureHeader.biWidth & 0x03;
		unsigned char padding_buffer[3] = { 0 };

		for (int i = 0; i < pictureHeader.biHeight; i++)
		{
			fwrite(pixelTab->data() + i * (size_t)pictureHeader.biWidth, sizeof(Pixel), pictureHeader.biWidth, file);
			fwrite(padding_buffer, sizeof(unsigned char), padding, file);
		}
		fclose(file);
	}
	return 0;
}
void BMPImage::ArrayToSOA()
{
	PixelSOA.blue = std::unique_ptr<uint8_t>(new uint8_t[pixelTab->size()]);
	PixelSOA.green = std::unique_ptr<uint8_t>(new uint8_t[pixelTab->size()]);
	PixelSOA.red = std::unique_ptr<uint8_t>(new uint8_t[pixelTab->size()]);
	PixelSOA.N = pixelTab->size();

	for (int i = 0; i < pixelTab->size(); ++i)
	{
		PixelSOA.blue.get()[i] = (*pixelTab)[i].blue;
		PixelSOA.green.get()[i] = (*pixelTab)[i].green;
		PixelSOA.red.get()[i] = (*pixelTab)[i].red;
	}
}
void BMPImage::ReadBMPasSOA(std::string filename)
{
	errno_t err;
	FILE* file;
	if ((err = fopen_s(&file, filename.c_str(), "rb")) != 0)
	{
		char buf[95] = { '\0' };
		strerror_s(buf, sizeof buf, err);
		fprintf_s(stderr, "cannot open file '%s': %s\n",
			filename.c_str(), buf);
	}
	else
	{
		fread(&fileHeader, sizeof(BMPFileHeader), 1, file);
		fseek(file, 14, SEEK_SET);
		fread(&pictureHeader, sizeof(BMPPictureHeader), 1, file);

		uint8_t padding = pictureHeader.biWidth & 0x03;

		PixelSOA.N = (long long)pictureHeader.biWidth * (long long)pictureHeader.biHeight;
		PixelSOA.blue = std::unique_ptr<uint8_t>(new uint8_t[PixelSOA.N]);
		PixelSOA.green = std::unique_ptr<uint8_t>(new uint8_t[PixelSOA.N]);
		PixelSOA.red = std::unique_ptr<uint8_t>(new uint8_t[PixelSOA.N]);

		fseek(file, fileHeader.bfOffBits, SEEK_SET);

		for (int i = 0; i < pictureHeader.biHeight; i++)
		{
			for (int j = 0; j < pictureHeader.biWidth; j++)
			{
				fread(PixelSOA.blue.get() + i * pictureHeader.biWidth + j, sizeof(uint8_t), 1, file);
				fread(PixelSOA.green.get() + i * pictureHeader.biWidth + j, sizeof(uint8_t), 1, file);
				fread(PixelSOA.red.get() + i * pictureHeader.biWidth + j, sizeof(uint8_t), 1, file);
			}
			fseek(file, padding, SEEK_CUR);
		}
		fclose(file);
	}
}

