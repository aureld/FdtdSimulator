// FdtdCoreTests.cpp : Defines the entry point for the console application.
//


#include <tiffio.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char* argv[])
{
	srand(time(NULL));
	//read test
	TIFF* tif = TIFFOpen("samples/foo.tif", "r");
	TIFFClose(tif);

	//write test
	tif = TIFFOpen("samples/bar.tif", "w");

	int sampleperpixel = 1;
	int width = 128;
	int height = 128;
	char *image = new char[width*height*sampleperpixel];

	for (int i = 0; i < width*height*sampleperpixel; i++)
	{
		image[i] = i;
	}

	TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);  // set the width of the image
	TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);    // set the height of the image
	TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, sampleperpixel);   // set number of channels per pixel
	TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);    // set the size of the channels
	TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);    // set the origin of the image.
	//TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
	TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, (sampleperpixel > 2) ? PHOTOMETRIC_RGB : PHOTOMETRIC_MINISBLACK);

	
	unsigned char *buf = NULL;        // buffer used to store the row of pixel information for writing to file
	//    Allocating memory to store the pixels of current row
	buf = (unsigned char *)_TIFFmalloc(width*height*sampleperpixel);

	for (int i = 0; i < width*height; i++) /* Grayscale */
		buf[i] = 128;
	TIFFWriteEncodedStrip(tif, 0, buf, width * height * sampleperpixel);
	TIFFClose(tif);

}

