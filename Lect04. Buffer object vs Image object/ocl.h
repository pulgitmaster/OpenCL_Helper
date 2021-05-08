#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
// OpenCL Headers
#include "CL/cl.hpp"
#include <iostream>
#include <fstream>
#include <vector>

using std::vector;

extern char					platformChooser;
extern vector<cl::Platform> platforms;
extern cl::Context			context;
extern cl::CommandQueue		commandQueue;
extern cl::Program			program;

extern cl::Kernel			gaussian_buffer_32FC1Kernel, gaussian_img_32FC1Kernel;

extern cl_int				errNum;
void ocl_init();