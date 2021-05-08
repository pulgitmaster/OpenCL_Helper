#include "ocl.h"
#include <cctype>

char					platformChooser;
vector<cl::Platform>	platforms;
cl::Context				context;
cl::CommandQueue		commandQueue;
cl::Program				program;

cl::Kernel				gaussian_buffer_32FC1Kernel, gaussian_img_32FC1Kernel;

vector<cl::Device>		devices;
cl_int					errNum;

void ocl_init()
{
	// query for platform
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		printf("Platform size 0\n");
		return;
	}

	// Get the number of platform and information about the platform
	printf("Platform number is : %d\n", (int)platforms.size());
	std::string platformVendor;

	for (unsigned int i = 0; i < platforms.size(); ++i) {
		platforms[i].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
		printf("Platform is by : %s\n", platformVendor.c_str());
		if (std::toupper(platformVendor.at(0)) == 'A' || std::toupper(platformVendor.at(0)) == 'N') // Only choose AMD or NVIDIA or ARM
		{
			platformChooser = i; break;
		}
	}

	// After choose platform save it property(generally choose 0)
	cl_context_properties properties[] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)(platforms[platformChooser])(),		// platforms[1] for laptop, platform [0] for desktop in my case...
		0
	};

	// 위의 properties를 사용하여 context 생성
	context = cl::Context(CL_DEVICE_TYPE_GPU, properties);

	// 플랫폼의 디바이스 정보 얻어옴
	devices = context.getInfo<CL_CONTEXT_DEVICES>();
	printf("Device number is : %d\n", (int)devices.size());	// device 번호, 이름 출력
	for (unsigned int i = 0; i < devices.size(); ++i) {
		printf("Device #%d : %s\n", i, devices[i].getInfo<CL_DEVICE_NAME>().c_str());
	}

	cl_int err = CL_SUCCESS;

	// Generate first command queue for device1
	printf("making command queue for device[0]\n");
	commandQueue = cl::CommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &err); // 3rd parameter 

	std::ifstream sourceFile("kernel.cl");
	std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
	cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));
	cl::Program program = cl::Program(context, source);
	program.build(devices, "-cl-fast-relaxed-math");
	while (program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) != CL_BUILD_SUCCESS) {
		printf("[build err at kernel.cl]  %s\n", program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]).c_str());
		return;
	}

	// create Kernel
	gaussian_buffer_32FC1Kernel = cl::Kernel(program, "gaussian_buffer_32FC1", &err);
	gaussian_img_32FC1Kernel = cl::Kernel(program, "gaussian_img_32FC1", &err);

}

