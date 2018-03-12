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
#include <time.h>
#include <ctime>


int main()
{
	/* Size of Matrix A : [3f x 4f], Size of Matrix B : [4f x 3f] (f means factor)*/
	const int factor = 16;
	const int heightA = 3*factor;
	const int widthA = 4*factor;
	const int heightB = 4*factor;
	const int widthB = 3*factor;
	int widthC=widthB, heightC=heightA;

	/* Matrix A, B declaration */
	float *A = new float[heightA * widthA];
	float *B = new float[heightB * widthB];
	float *C = new float[heightC * widthC];

	/* Matrix A, B initilization  */
	for (int i = 0; i < heightA * widthA; i++) {
		A[i] = (float)(i % 10) + 0.1f;
	}
	for (int i = 0; i < heightB * widthB; i++) {
		B[i] = (float)(i % 10) + 0.1f;
	}

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
			(cl_context_properties)(platforms[0])(),
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

		// 메모리 버퍼의 생성 A, B, C를 선언하고 초기화 된 float배열이라 가정한다.
		cl::Buffer bufferA = cl::Buffer(context, CL_MEM_READ_ONLY, widthA * heightA * sizeof(float));
		cl::Buffer bufferB = cl::Buffer(context, CL_MEM_READ_ONLY, widthB * heightB * sizeof(float));
		cl::Buffer bufferC = cl::Buffer(context, CL_MEM_WRITE_ONLY, heightC * widthC * sizeof(float));

		// 첫 번째 디바이스를 위한 명령어 큐를 사용하여 입력 버퍼에 입력 데이터를 복사
		queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, widthA * heightA * sizeof(float), A);
		queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, widthB * heightB * sizeof(float), B);

		// 프로그램 소스를 읽어옴
		std::ifstream sourceFile("vector_matmul_kernel.cl");
		std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
		cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));	// 재활용성이 높은 고 수준의 프로그래밍이 가능함 in CPP !

																										// 소스 코드로부터 프로그램을 생성
		cl::Program program = cl::Program(context, source);

		// 디바이스를 위한 프로그램을 빌드
		program.build(devices);

		// 커널을 생성
		cl::Kernel kernel(program, "matmul", &err);	// 커널함수 명을 parameter로 전달해주면 됨.

													// 이벤트 생성 ---> 아직 용도를 몰겠음.
		cl::Event event;

		// 커널 매개변수를 설정
		kernel.setArg(0, bufferC);
		kernel.setArg(1, widthA);
		kernel.setArg(2, heightA);
		kernel.setArg(3, widthB);
		kernel.setArg(4, heightB);
		kernel.setArg(5, bufferA);
		kernel.setArg(6, bufferB);

		// 지역/전역 워크그룹 크기를 설정
		// 행렬 크기가 16(=factor)으로 나누어 떨어진다고 가정.
		cl::NDRange localws(factor, factor);
		cl::NDRange globalws(widthC, heightC);

		// GPU와 CPU 계산 시간을 재 봅시다.
		/*std::cout << "\nWork GROUP : " << workGR << std::endl;
		std::cout << "Work ID : " << workID << std::endl;
		std::cout << "total array size : " << N_ELEMENTS << std::endl;*/

		// 커널을 실행
		//cl::NDRange global(N_ELEMENTS);
		//cl::NDRange local(workID);	// 워크 그룹당 워크아이디 256개

									// 먼저 GPU 계산 부터~
		//std::cout << "\n----- GPU calculation start -----" << std::endl;
		//clock_t gpu_begin = clock();	// GPU계산 시작
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalws, localws, NULL, &event);
		event.wait();	// 뭥미?

		queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, heightC * widthC * sizeof(int), C);	// 출력 데이터를 호스트로 다시 복사

		//clock_t gpu_end = clock();		// GPU계산 끝
		//double gpu_elapsed_secs = ((double)gpu_end - (double)gpu_begin) / CLOCKS_PER_SEC;
		/*std::cout << "cost time : " << gpu_elapsed_secs << std::endl;
		std::cout << "----- GPU calculation end -----" << std::endl;*/

		// 두 번째로 CPU 계산~
		//std::cout << "----- CPU calculation start -----" << std::endl;
		//clock_t cpu_begin = clock();	// CPU 계산 시작
		//for (int i = 0; i < N_ELEMENTS; i++) {
		//	CC[i] = A[i] + B[i];
		//}
		//clock_t cpu_end = clock();		// CPU 계산 끝
		//double cpu_elapsed_secs = ((double)cpu_end - (double)cpu_begin) / CLOCKS_PER_SEC;
		//std::cout << "cost time : " << cpu_elapsed_secs << std::endl;
		//std::cout << "----- CPU calculation end -----" << std::endl;

		// GPU 계산 결과 검증
		bool result = true;
		for (int i = 0; i < heightC * widthC; i++) {
			std::cout << C[i] << " ";
		}
		std::cout << std::endl;
		/*if (result) {
			std::cout << "OpenCL Calculation correct!" << std::endl;
		}
		else {
			std::cout << "OpenCl Calculation failed!" << std::endl;
		}*/
	}
	catch (cl::Error err)	// 예외 처리 구문 try의 err변수를 이용하여 err 출력
	{
		std::cerr << "Error: " << err.what() << "(" << err.err() << ")" << std::endl;
	}

	return EXIT_SUCCESS;
}

