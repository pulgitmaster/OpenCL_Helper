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

	/* A, B 배열 초기화 -- 0~N_ELEMENTS 값으로 초기화 시키기 */
	for (int i = 0; i < N_ELEMENTS; i++) {
		A[i] = i;
		B[i] = i;
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

		// 이건 잘 모르겟... properties? 뭔가 매개변수로 넘겨주는데 역할을 잘 모르겠...
		cl_context_properties properties[] =
		{
			CL_CONTEXT_PLATFORM,
			(cl_context_properties)(platforms[0])(),	// platforms[x]에 데스크톱에서는 gpu가 1개만 인식 --> x = 0, 노트북에선 2개 인식 --> x = 1
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

		// 메모리 버퍼의 생성
		cl::Buffer bufferA = cl::Buffer(context, CL_MEM_READ_ONLY, N_ELEMENTS * sizeof(int));
		cl::Buffer bufferB = cl::Buffer(context, CL_MEM_READ_ONLY, N_ELEMENTS * sizeof(int));
		cl::Buffer bufferC = cl::Buffer(context, CL_MEM_WRITE_ONLY, N_ELEMENTS * sizeof(int));

		// 첫 번째 디바이스를 위한 명령어 큐를 사용하여 입력 버퍼에 입력 데이터를 복사
		queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, N_ELEMENTS * sizeof(int), A);
		queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, N_ELEMENTS * sizeof(int), B);

		// 프로그램 소스를 읽어옴
		std::ifstream sourceFile("vector_add_kernel.cl");
		std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
		cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));	// 재활용성이 높은 고 수준의 프로그래밍이 가능함 in CPP !

		// 소스 코드로부터 프로그램을 생성
		cl::Program program = cl::Program(context, source);

		// 디바이스를 위한 프로그램을 빌드
		program.build(devices);

		// 커널을 생성
		cl::Kernel kernel(program, "vecadd", &err);	// 커널함수 명을 parameter로 전달해주면 됨.

		// 이벤트 생성 ---> 아직 용도를 몰겠음.
		cl::Event event;

		// 커널 매개변수를 설정
		kernel.setArg(0, bufferA);
		kernel.setArg(1, bufferB);
		kernel.setArg(2, bufferC);

		// 커널을 실행
		cl::NDRange global(N_ELEMENTS);
		cl::NDRange local(256);	// 워크 그룹당 워크아이디 256개
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, NULL, &event);
		event.wait();	// 뭥미?

		// 출력 데이터를 호스트로 다시 복사
		queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, N_ELEMENTS * sizeof(int), C);

		// 결과를 검증
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
	catch (cl::Error err)	// 예외 처리 구문 try의 err변수를 이용하여 err 출력
	{
		std::cerr << "Error: " << err.what() << "(" << err.err() << ")" << std::endl;
	}

    return EXIT_SUCCESS;
}

