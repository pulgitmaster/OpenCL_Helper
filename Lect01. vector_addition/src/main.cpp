#define __CL_ENABLE_EXCEPTIONS

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.hpp>
#endif

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

int main()
{
	const int N_ELEMENTS = 2048;
	int *A = new int[N_ELEMENTS];
	int *B = new int[N_ELEMENTS];
	int *C = new int[N_ELEMENTS];

	for (int i = 0; i < N_ELEMENTS; i++) {
		A[i] = i;
		B[i] = i;
	}

	const char* kernel_path = { "../kernel/vector_add_kernel.cl" };
	cl_int err = CL_SUCCESS;
	try
	{
		std::vector<cl::Platform>platforms;
		cl::Platform::get(&platforms);
		if (platforms.size() == 0) {
			std::cout << "Platform size 0\n";
			return -1;
		}

		std::cout << "Platform number is: " << platforms.size() << std::endl;
		std::string platformVendor;
		for (unsigned int i = 0; i < platforms.size(); ++i) {
			platforms[i].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
			std::cout << "Platform is by: " << platformVendor << std::endl;
		}

		cl_context_properties properties[] =
		{
			CL_CONTEXT_PLATFORM,
			(cl_context_properties)(platforms[0])(),
			0
		};

		cl::Context context(CL_DEVICE_TYPE_ALL, properties);

		std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
		std::cout << "Device number is: " << devices.size() << std::endl;
		for (unsigned int i = 0; i < devices.size(); ++i) {
			std::cout << "Device #" << i << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
		}

		cl::CommandQueue queue(context, devices[0], 0, &err);

		cl::Buffer bufferA = cl::Buffer(context, CL_MEM_READ_ONLY, N_ELEMENTS * sizeof(int));
		cl::Buffer bufferB = cl::Buffer(context, CL_MEM_READ_ONLY, N_ELEMENTS * sizeof(int));
		cl::Buffer bufferC = cl::Buffer(context, CL_MEM_WRITE_ONLY, N_ELEMENTS * sizeof(int));

		queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, N_ELEMENTS * sizeof(int), A);
		queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, N_ELEMENTS * sizeof(int), B);

		std::ifstream sourceFile(kernel_path);
		std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
		cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));

		cl::Program program = cl::Program(context, source);

		program.build(devices);

		cl::Kernel kernel(program, "vecadd", &err);

		cl::Event event;

		kernel.setArg(0, bufferA);
		kernel.setArg(1, bufferB);
		kernel.setArg(2, bufferC);

		cl::NDRange global(N_ELEMENTS);
		cl::NDRange local(256);
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, NULL, &event);
		event.wait();

		queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, N_ELEMENTS * sizeof(int), C);

		bool result = true;
		for (int i = 0; i < N_ELEMENTS; i++) {
			if (C[i] != A[i] + B[i]) {
				result = false;
				break;
			}
		}

		if (result) {
			std::cout << "Success!" << std::endl;
		}
		else {
			std::cout << "Failed!" << std::endl;
		}
	}
	catch (cl::Error err)
	{
		std::cerr << "Error: " << err.what() << "(" << err.err() << ")" << std::endl;
	}

    return EXIT_SUCCESS;
}

