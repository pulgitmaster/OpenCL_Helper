#include "ocl.h"
#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;

void gaussian_buffer();
void gaussian_image_object();
void filterCreation1D(float* GKernel, int kSize, double sigma);
double executionTime(cl::Event event);

int main(int argc, char ** argv) {
	ocl_init();

	gaussian_image_object();
	//gaussian_buffer();
	return 0;
}

void gaussian_buffer()
{
	// Load image
	Mat src, dst;
	int width, height;
	src = imread("test.jpg", 0); // gray
	width = src.cols; height = src.rows;

	cl::Buffer dev_ImgSrc = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uchar)*width*height, src.data);
	cl::Buffer dev_ImgDst = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*width*height);

	// Create Gaussian filter mask
	int kSize = 5;
	int mSize = kSize / 2;
	float *gKernel = new float[kSize * kSize];
	filterCreation1D(gKernel, kSize, 1.5);

	// Create buffer for mask and transfer it to the device
	cl::Buffer dev_gKernel = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* kSize * kSize, gKernel);

	// Run Gaussian kernel
	cl::Event gaussian_buffer_32FC1_evt;
	gaussian_buffer_32FC1Kernel.setArg(0, dev_ImgSrc);
	gaussian_buffer_32FC1Kernel.setArg(1, dev_ImgDst);
	gaussian_buffer_32FC1Kernel.setArg(2, dev_gKernel);
	gaussian_buffer_32FC1Kernel.setArg(3, width);
	gaussian_buffer_32FC1Kernel.setArg(4, mSize);

	commandQueue.enqueueNDRangeKernel(gaussian_buffer_32FC1Kernel, cl::NDRange(mSize, mSize), cl::NDRange(width - 2 * mSize, height - 2 * mSize), cl::NullRange, NULL, &gaussian_buffer_32FC1_evt);
	gaussian_buffer_32FC1_evt.wait();

	// Transfer buffer back to host
	float* data = new float[width * height];
	commandQueue.enqueueReadBuffer(dev_ImgDst, CL_TRUE, 0, sizeof(float)*width * height, data);
	printf("Execution time(%s) : %.5lfms\n", "gaussian_buffer_32FC1_evt", executionTime(gaussian_buffer_32FC1_evt));

	dst = Mat(height, width, CV_32FC1, data);

	// Convert float channel to uint8
	Mat dst2;
	dst.convertTo(dst2, CV_8UC1);
	imwrite("gaussian_buffer.jpg", dst2);
}

void gaussian_image_object()
{
	// Load image
	Mat src, dst;
	int width, height;
	src = imread("test.jpg", 0); // gray
	width = src.cols; height = src.rows;

	// Create an OpenCL Image / texture and transfer data to the device
	cl::Image2D dev_ImgSrc;
	try {
		dev_ImgSrc = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, cl::ImageFormat(CL_R, CL_UNSIGNED_INT8), width, height, /*length of row*/sizeof(uchar)*width, src.data, &errNum);
		// ref : https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/cl_image_format.html
	}
	catch (cl::Error& e)
	{
		std::cout << e.what() << "(" << e.err() << ")" << "\n";
	}

	// Create a buffer for the result
	cl::Image2D dev_ImgDst = cl::Image2D(context, CL_MEM_WRITE_ONLY, cl::ImageFormat(CL_INTENSITY, CL_FLOAT), width, height, 0, NULL, &errNum);

	// Create Gaussian filter mask
	int kSize = 5;
	float *gKernel = new float[kSize * kSize];
	filterCreation1D(gKernel, kSize, 1.5);

	// Create buffer for mask and transfer it to the device
	cl::Buffer dev_gKernel = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* kSize * kSize, gKernel);

	// Run Gaussian kernel
	cl::Event gaussian_img_32FC1_evt;
	gaussian_img_32FC1Kernel.setArg(0, dev_ImgSrc);
	gaussian_img_32FC1Kernel.setArg(1, dev_ImgDst);
	gaussian_img_32FC1Kernel.setArg(2, dev_gKernel);
	gaussian_img_32FC1Kernel.setArg(3, kSize / 2);

	commandQueue.enqueueNDRangeKernel(gaussian_img_32FC1Kernel,	cl::NullRange,	cl::NDRange(width, height),	cl::NullRange, NULL, &gaussian_img_32FC1_evt);
	gaussian_img_32FC1_evt.wait();

	// Transfer image back to host
	float* data = new float[width * height];
	cl::size_t<3> origin; origin[0] = 0; origin[1] = 0; origin[2] = 0;
	cl::size_t<3> region; region[0] = width; region[1] = height; region[2] = 1;
	commandQueue.enqueueReadImage(dev_ImgDst, CL_TRUE, origin, region, sizeof(float)*width, 0, data);
	printf("Execution time(%s) : %.5lfms\n", "gaussian_img_32FC1_evt", executionTime(gaussian_img_32FC1_evt));

	dst = Mat(height, width, CV_32FC1, data);

	// Convert float channel to uint8
	Mat dst2;
	dst.convertTo(dst2, CV_8UC1);
	imwrite("gaussian_image.jpg", dst2);
}

// Function to create Gaussian filter 
void filterCreation1D(float* GKernel, int kSize, double sigma)
{
	// intialising standard deviation to 1.0 
	double r, s = 2.0 * sigma * sigma;

	// sum is for normalization 
	float sum = 0.f;

	int m_size = (kSize / 2);

	// generating kSzie x kSize kernel 
	for (int x = -m_size; x <= m_size; x++) {
		for (int y = -m_size; y <= m_size; y++) {
			r = (x * x + y * y);
			GKernel[(x + m_size)*(kSize)+(y + m_size)] = static_cast<float>((exp((-r) / s)) / (CV_PI * s));
			sum += GKernel[(x + m_size)*(kSize)+(y + m_size)];
		}
	}

	// normalizing the Kernel 
	for (int i = 0; i < kSize; i++)
		for (int j = 0; j < kSize; j++)
			GKernel[i * kSize + j] /= sum;
}

double executionTime(cl::Event event)
{
	cl_ulong start, end;

	start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

	return (double)1.0e-6 * (double)(end - start); // convert nanoseconds to milli-seconds on return
}