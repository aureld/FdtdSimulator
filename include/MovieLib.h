// movielib.h : Movie related function using VTK and FFMPEG
// Aurelien Duval 2015
#pragma once
#ifndef _MOVIELIB_H_
#define _MOVIELIB_H_



#include <string>

class vtkImageData;
class vtkFFMPEGWriter;

class Movie {


public:
	Movie();
	~Movie();

	bool Initialize(int sizeX, int sizeY);
	void SetFileName(const char* filename) { this->filename = filename; }
	bool Start(); //starts the movie
	void SetData(unsigned char* data); //writes data into buffer
	void SetOverlayStepNumber(int time); //writes the timestep # in the frame
	bool Write(); //writes a frame
	bool End(); //ends the movie
	

private:
	// internal data storage
	vtkImageData* imageData;
	vtkFFMPEGWriter* movieWriter;
	
	// extents
	int sizeX;
	int sizeY;

	std::string filename; //movie filenmae
	
};

#endif //_MOVIELIB_H_