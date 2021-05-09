#ifndef _OCL_HPP_
#define _OCL_HPP_

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
using std::ifstream;
using std::string;
using std::vector;

class OCL
{
private:
    vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    cl::Program program;
    bool initialized;

public:
    cl::Context context;
    cl::CommandQueue cmdQueue;

public:
    OCL();
    OCL(int platform_num, const char *device_type, int device_num);
    ~OCL();
    bool Initialized();
    int BuildProgram(const char *kernel_source_path);
    int BuildProgram(const char *kernel_source_path, int device_num);
    int CreateKernel(cl::Kernel &kernel, const char *kernel_function_name);
    void GetDevicesInfo();
    void GetDeviceInfo(int device_num);
};

#endif // _OCL_HPP_