// runTester.cpp : Defines the entry point for the console application.
//
#include "MovieLib.h"

int main()
{
	COW_MovieEngine *movie = new COW_MovieEngine;
	movie->SetFileName("test.avi");
	movie->SetFrameRate(30);
	movie->SetOutputSize(256, 256);
	movie->Initialize(256, 256);
	movie->StartMovieFFMPEG();

	unsigned char *data = new unsigned char [movie->GetDataSize()];

	for (int t = 0; t < 200; t++)
	{
		for (int i = 0; i < movie->GetDataSize(); i++)
			data[i] = i*t % 255;

		movie->SetData(data, 256, 256);
		movie->WriteMovie();
	}
	

	movie->EndMovie();
	return 0;
}

