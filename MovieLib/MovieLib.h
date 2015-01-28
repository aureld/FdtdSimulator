// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the OW_MOVIEENGINEDLL_EXPORTS
// symbol defined on the command line. this symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// OW_MOVIEENGINEDLL_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#pragma once
#ifndef _MOVIELIB_H_
#define _MOVIELIB_H_



#include <string>

//mp1042// mp2013 Sep 25 - START --- Improving movie export capabilities
//mp1042//class vtkAVIWriter;
class vtkGenericMovieWriter;
//mp1042// mp2013 Sep 25 - END --- Improving movie export capabilities
class vtkJPEGWriter;
class vtkImageData;
class vtkImageResize;

// This class is exported from the OW_MovieEngineDLL.dll
class COW_MovieEngine {
public:
	COW_MovieEngine();
	~COW_MovieEngine();

	// NOTE: Whenever you want to change the input size, you have to reinitialize
	// NOTE: Whenever you want to change the resize settings, you have to reinitialize
	// initializes the internals
	bool Initialize(int zn_data_size_x, int zn_data_size_y);
	// initializes the internals (thread-safe version)
	bool Initialize_Safe(int zn_data_size_x, int zn_data_size_y);

	// specify the filename of the movie
	// - note that the file will get overwritten but not completely 
	//TDO//    fix this internally?
	void SetFileName(const char* zps_filename) { this->mstrFilename = zps_filename; }

	// specify the frame rate of the movie
	// - each call to Write() equals one frame of the movie
	void SetFrameRate(int zn_fps) { this->mnFrameRate = zn_fps; }

	// specify the quality of the movie
	// - 0 = lowest quality, 2 = highest quality
	void SetQuality(int zn_quality) { this->mnQuality = zn_quality; }

	// the output size if rescaling is used
	void EnableRescale(bool zb_rescale) { this->mbRescale = zb_rescale; }
	inline void SetOutputSize(int zn_movie_size_x, int zn_movie_size_y)
	{
		this->mnOutputSizeX = zn_movie_size_x;
		this->mnOutputSizeY = zn_movie_size_y;
	}

	// setting of data 
	// Method 1: pass in data to copy internally
	// - a check is made if the passed image size is equal to the internal storage
	void SetData(unsigned char* zarr_data, int zn_data_size_x, int zn_data_size_y);
	// Method 2: get data pointer and copy externally
	// get the pointer to the internal buffer to be modifed externally in an efficient manner
	// - the user has to make sure that they write within the limits
	// - the auxillary GetDataSize methods are provided to help the user
	int GetDataSize() const { return mnInputSizeX*mnInputSizeY * 3; }
	unsigned char* GetDataPtr();

	// NOTE: Whenever you want to change the movie settings, including the filename 
	//       .. you have to end the previous movie and start a new one
	// starts the movie engine and opens the movie data file
	bool StartMovieAVI();
	// starts the movie engine and opens the movie data file (thread-safe version)
	bool StartMovieAVI_Safe();

	//mp1042// mp2013 Sep 25 - START --- Improving movie export capabilities
	/// Start making an FFMPEG movie
	bool StartMovieFFMPEG();
	bool StartMovieFFMPEG_Safe();
	//mp1042// mp2013 Sep 25 - END --- Improving movie export capabilities

	// writes internal data to the movie engine from internal buffer
	bool WriteMovie();
	// writes internal data to the movie engine from internal buffer (thread-safe version)
	// ... Note that using this function might slow down the threads since they
	// ... will have to wait for each other so use when necessary
	bool WriteMovie_Safe();

	// finishes the movie engine and closes the movie data file
	bool EndMovie();
	// writes internal data to the movie engine from internal buffer (thread-safe version)
	bool EndMovie_Safe();

	// NOTE: before you call the WriteImage function, you need to run Initialize
	// saves the buffer into an image format
	bool WriteImage(const char* zps_filename);

	// this is how to check if the movie is being writen
	bool IsInProgress() { return mbMovieInProgress; }

private:
	void Cleanup();
	void SetupFilters();

	// this is how to check for rescaling - do not use the rescale flag directly
	bool IsRescaled() { return NULL != mpovtkResizeFilter; }

	// internal data storage
	vtkImageData* mpovtkImageData;

	// the movie writer using the Windows AVI library
	//mp1042// mp2013 Sep 25 - START --- Improving movie export capabilities
	//mp1042//    vtkAVIWriter* mpoMovieWriter;
	vtkGenericMovieWriter* mpoMovieWriter;
	//mp1042// mp2013 Sep 25 - END --- Improving movie export capabilities

	// the image writer using JPEG
	vtkJPEGWriter* mpoImageWriter;
	// filter for resizing the image data
	vtkImageResize* mpovtkResizeFilter;

	// a movie is currently being made
	bool mbMovieInProgress;

	// stored data parameters
	int mnInputSizeX;
	int mnInputSizeY;

	// the frame rate (frames per second)
	int mnFrameRate;
	// the quality (values 0 - 2, 2 being highest)
	int mnQuality;
	// the filename of the output
	std::string mstrFilename;

	// NOTE: the mbRescale flag is only used to denote rescaling at initialization time
	//       .. it should not be used to query rescaling since the user can set it at
	//       .. any time - use the IsRescaled() method for query.
	bool mbRescale;
	// resized movie output size
	int mnOutputSizeX;
	int mnOutputSizeY;
};

#endif //_MOVIELIB_H_