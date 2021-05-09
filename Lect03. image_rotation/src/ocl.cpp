#include "ocl.hpp"
#include <cstring>

OCL::OCL()
{
    initialized = false;

    cl_int err = CL_SUCCESS;

    // Get platforms able to run OpenCL
    cl::Platform::get(&platforms);
    if (platforms.size() == 0)
    {
        std::cerr << "[ERROR]Platform size 0\n";
        return;
    }

    // Get context for all device types
    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0}; // platform 0 is default
    context = cl::Context(CL_DEVICE_TYPE_ALL, properties);

    // Get devices from context
    devices = context.getInfo<CL_CONTEXT_DEVICES>();

    cmdQueue = cl::CommandQueue(context, devices[0], 0, &err); // device 0 is default
    if (err < CL_SUCCESS /*=0*/)
    {
        std::cerr << "[ERROR]clCreateCommandQueue error code: " << err << "\n";
        return;
    }

    initialized = true;
}

OCL::OCL(int platform_num, const char *device_type, int device_num)
{
    initialized = false;

    cl_int err = CL_SUCCESS;

    // Get platforms able to run OpenCL
    cl::Platform::get(&platforms);
    if (platforms.size() == 0)
    {
        std::cerr << "[ERROR]Platform size 0\n";
        return;
    }

    if (platforms.size() <= platform_num)
    {
        std::cerr << "[ERROR]Invalid platform_num, platforms size: " << platforms.size() << "\n";
        return;
    }

    // Get context for all device types
    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[platform_num])(), 0}; // platform 0 is default
    if (strncmp(device_type, "CPU", 3) == 0 || strncmp(device_type, "cpu", 3) == 0)
    {
        context = cl::Context(CL_DEVICE_TYPE_CPU, properties);
    }
    else if (strncmp(device_type, "GPU", 3) == 0 || strncmp(device_type, "gpu", 3) == 0)
    {
        context = cl::Context(CL_DEVICE_TYPE_GPU, properties);
    }
    else
    {
        context = cl::Context(CL_DEVICE_TYPE_ALL, properties);
    }

    // Get devices from context
    devices = context.getInfo<CL_CONTEXT_DEVICES>();

    if (devices.size() <= device_num)
    {
        std::cerr << "[ERROR]Invalid device_num, devices size: " << devices.size() << "\n";
        return;
    }

    cmdQueue = cl::CommandQueue(context, devices[device_num], 0, &err); // device 0 is default
    if (err < CL_SUCCESS /*=0*/)
    {
        std::cerr << "[ERROR]clCreateCommandQueue error code: " << err << "\n";
        return;
    }

    initialized = true;
}

OCL::~OCL()
{
    platforms.clear();
    devices.clear();
    initialized = false;
}

bool OCL::Initialized()
{
    return initialized;
}

int OCL::BuildProgram(const char *kernel_source_path)
{
    ifstream source_file(kernel_source_path);
    string source_code(std::istreambuf_iterator<char>(source_file), (std::istreambuf_iterator<char>()));
    cl::Program::Sources sources(1, std::make_pair(source_code.c_str(), source_code.length() + 1));
    program = cl::Program(context, sources);
    try
    {
        program.build(devices);
    }
    catch (cl::Error &e)
    {
        if (e.err() == CL_BUILD_PROGRAM_FAILURE)
        {
            for (cl::Device device : devices)
            {
                // Check the build status
                cl_build_status status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device);
                if (status != CL_BUILD_ERROR)
                    continue;

                // Get the build log
                std::string name = device.getInfo<CL_DEVICE_NAME>();
                std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                std::cerr << "Build log for " << name << ":" << std::endl
                          << buildlog << std::endl;
            }
			return -1;
        }
        else
        {
            throw e;
        }
    }
    return CL_SUCCESS;
}

int OCL::BuildProgram(const char *kernel_source_path, int device_num)
{
    if (devices.size() <= device_num)
    {
        std::cerr << "[ERROR]Invalid device_num, devices size: " << devices.size() << "\n";
        return -1;
    }

	ifstream source_file(kernel_source_path);
	string source_code(std::istreambuf_iterator<char>(source_file), (std::istreambuf_iterator<char>()));
	cl::Program::Sources sources(1, std::make_pair(source_code.c_str(), source_code.length() + 1));
    program = cl::Program(context, sources);
    try
    {
        program.build(devices);
    }
    catch (cl::Error &e)
    {
        if (e.err() == CL_BUILD_PROGRAM_FAILURE)
        {
            cl::Device device = devices[device_num];

            // Check the build status
            cl_build_status status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device);
			if (status == CL_BUILD_ERROR) {
				// Get the build log
				std::string name = device.getInfo<CL_DEVICE_NAME>();
				std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
				std::cerr << "Build log for " << name << ":" << std::endl
					<< buildlog << std::endl;
			}
			return -1;
        }
        else
        {
            throw e;
        }
    }
    return CL_SUCCESS;
}

int OCL::CreateKernel(cl::Kernel &kernel, const char *kernel_function_name)
{
    cl_int err = CL_SUCCESS;
    kernel = cl::Kernel(program, kernel_function_name, &err);
    if (err < CL_SUCCESS)
    {
        std::cerr << "[ERROR]clCreateKernel error code: " << err << "\n";
    }
	return err;
}

void OCL::GetDevicesInfo()
{
    std::cout << "Total number of device in this platform is : " << devices.size() << "\n\n";
    for (int i = 0; i < devices.size(); ++i)
    {
        std::vector<size_t> max_work_items = devices[i].getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
        uint32_t max_work_items_dims = devices[i].getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
        std::cout << "  Device #" << i << std::endl
                  << "  Vendor : " << devices[i].getInfo<CL_DEVICE_VENDOR>() << std::endl
                  << "  Name : " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl
                  << "  Device Version : " << devices[i].getInfo<CL_DEVICE_VERSION>() << std::endl
                  << "  OpenCL version : " << devices[i].getInfo<CL_DEVICE_OPENCL_C_VERSION>() << std::endl
                  << "  Max Compute Units : " << devices[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl
                  << "  Max Work Item Dimensions : " << max_work_items_dims << std::endl
                  << "  Max Work Item Sizes" << std::endl;
        for (int j = 0; j < max_work_items_dims; j++)
            std::cout << "	- Work Item[" << j << "] Size : " << max_work_items[j] << std::endl << std::endl;
    }
    std::cout << std::endl;
}

void OCL::GetDeviceInfo(int device_num)
{
    std::cout << "print device[" << device_num << "] info\n";
    std::vector<size_t> max_work_items = devices[device_num].getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
    uint32_t max_work_items_dims = devices[device_num].getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
    std::cout << "  Vendor : " << devices[device_num].getInfo<CL_DEVICE_VENDOR>() << std::endl
              << "  Name : " << devices[device_num].getInfo<CL_DEVICE_NAME>() << std::endl
              << "  Device Version : " << devices[device_num].getInfo<CL_DEVICE_VERSION>() << std::endl
              << "  OpenCL version : " << devices[device_num].getInfo<CL_DEVICE_OPENCL_C_VERSION>() << std::endl
              << "  Max Compute Units : " << devices[device_num].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl
              << "  Max Work Item Dimensions : " << max_work_items_dims << std::endl
              << "  Max Work Item Sizes" << std::endl;
    for (int j = 0; j < max_work_items_dims; j++)
        std::cout << "	- Work Item[" << j << "] Size : " << max_work_items[j] << std::endl;
    std::cout << std::endl;
}