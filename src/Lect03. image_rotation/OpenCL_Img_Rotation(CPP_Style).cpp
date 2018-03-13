#define __CL_ENABLE_EXCEPTIONS

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.hpp>
#endif

#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <time.h>
#include <ctime>

// BMP uilities
#include "bmpfuncs.h"

/*
이미지 회전 변환을 위한 기초 지식

점(x1, y1)의 좌표가 (x0, y0) 기준으로 각도 θ만큼 회전되면 다음의 식과 같이 (x2, y2)가 된다.
x2 = cos(θ)*(x1-x0) + sin(θ)*(y1-y0)
y2 = -sin(θ)*(x1-x0) + cos(θ)*(y1-y0)

만약 기준 점이 원점이라면
x2 = cos(θ)*x1 + sin(θ)*y1
y2 = -sin(θ)*x1 + cos(θ)*y1
*/

#define PI 3.141592

int main()
{
	/* Size of Image : [W * H] */
	double degree = 60;	// degree만큼 회전
	double theta = (double) (PI / (double)(180 / degree));
	int W;
	int H;

	/* 같은 경로 내에 반드시 input.bmp가 존재해야한다. */
	const char* inputFile = "input.bmp";
	const char* outputFile = "output.bmp";

	/* input image ip, output image op */
	float *ip = readImage(inputFile, &W, &H);
	float *op = new float[W * H];

	cl_int err = CL_SUCCESS;

	try
	{
		// 플랫폼을 위한 쿼리
		std::vector<cl::Platform>platforms;
		cl::Platform::get(&platforms);
		if (platforms.size() == 0) {
			std::cout << "Platform size 0\n";
			return -1;
		}

		// 플랫폼의 숫자와 플랫폼 정보를 얻어옴
		std::cout << "Platform number is: " << platforms.size() << std::endl;
		std::string platformVendor;
		for (unsigned int i = 0; i < platforms.size(); ++i) {
			platforms[i].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
			std::cout << "Platform is by: " << platformVendor << std::endl;
		}

		// 플랫폼 선택 후 property 변수에 저장 (일반적인 단일 GPU일 경우 0번 플랫폼 선택)
		cl_context_properties properties[] =
		{
			CL_CONTEXT_PLATFORM,
			(cl_context_properties)(platforms[0])(),		// platforms[1] for laptop, platform [0] for desktop in my case...
			0
		};

		// 위의 properties를 사용하여 context 생성
		cl::Context context(CL_DEVICE_TYPE_ALL, properties);

		// 플랫폼의 디바이스 정보 얻어옴
		std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
		std::cout << "Device number is: " << devices.size() << std::endl;	// device 번호, 이름 출력
		for (unsigned int i = 0; i < devices.size(); ++i) {
			std::cout << "Device #" << i << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
		}

		// 첫 번째 디바이스를 위한 명령어 큐의 생성
		cl::CommandQueue queue(context, devices[0], 0, &err);

		// 메모리 버퍼의 생성한다. d_ip, d_op를 선언하고 초기화 된 float배열이라 가정한다.
		cl::Buffer d_ip = cl::Buffer(context, CL_MEM_READ_ONLY, W * H * sizeof(float));
		cl::Buffer d_op = cl::Buffer(context, CL_MEM_READ_ONLY, W * H * sizeof(float));

		// 첫 번째 디바이스를 위한 명령어 큐를 사용하여 입력 버퍼에 입력 데이터를 복사
		queue.enqueueWriteBuffer(d_ip, CL_TRUE, 0, W * H * sizeof(float), ip);

		// 프로그램 소스를 읽어옴
		std::ifstream sourceFile("img_rotation_kernel.cl");
		std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
		cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));	// 재활용성이 높은 고 수준의 프로그래밍이 가능함 in CPP !

		cl::Program program = cl::Program(context, source);		// 소스 코드로부터 프로그램을 생성

		// 디바이스를 위한 프로그램을 빌드
		program.build(devices);

		// 커널을 생성
		cl::Kernel kernel(program, "img_rotate", &err);	// 커널함수 명을 parameter로 전달해주면 됨.

		cl::Event event;	// event 생성 : 커널 함수의 현 상태 확인 시 매우 유용.

		// 회전 각도 θ에 대한 회전 인자 계산
		float cos_theta = cos(theta);
		float sin_theta = sin(theta);

		// 커널 매개변수를 설정
		kernel.setArg(0, d_op);
		kernel.setArg(1, d_ip);
		kernel.setArg(2, W);
		kernel.setArg(3, H);
		kernel.setArg(4, sin_theta);
		kernel.setArg(5, cos_theta);

		// 전역 워크그룹 크기를 설정
		// 이 번 예제에선 local work group size가 중요하지 않다
		// local work item과 local group 간의 통신하는 커널 소스가 없기 때문!
		cl::NDRange globalws(W, H);

		// kernel source 실행
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalws, cl::NullRange, NULL, &event);
		event.wait();	// kernel 소스가 끝날 때 까지 기다린다. 비동기 절차형 알고리즘에 매우 적합

		// 출력 버퍼를 호스트로 읽어온다.
		queue.enqueueReadBuffer(d_op, CL_TRUE, 0, W * H * sizeof(float), op);	// 출력 데이터를 호스트로 다시 복사

		storeImage(op, outputFile, H, W, inputFile);
	}
	catch (cl::Error err)	// 예외 처리 구문 try의 err변수를 이용하여 err 출력
	{
		std::cerr << "Error: " << err.what() << "(" << err.err() << ")" << std::endl;
	}

	delete ip, op;
	return EXIT_SUCCESS;
}

