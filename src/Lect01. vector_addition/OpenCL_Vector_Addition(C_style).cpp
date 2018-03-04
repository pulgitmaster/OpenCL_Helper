#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x1000)
int main()
{
	FILE *fp;
	char fileName[] = "vector_add.cl";
	/* 커널함수를 불러서 읽어온다. */
	fp = fopen(fileName, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	/* 정상적으로 불러온 커널함수에 대하여 메모리 할당 및 size 측정 */
	char *programSource = (char*)malloc(MAX_SOURCE_SIZE);
	size_t source_size = fread(programSource, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	// 호스트 데이터
	int *A = NULL;	// 입력 배열
	int *B = NULL;  // 입력 배열
	int *C = NULL;	// 출력 배열

	// 각 배열의 항목
	const int elements = 2048;

	// 데이터 크기 계산
	size_t datasize = sizeof(int)*elements;	// 뭐 결국 4bytes * 2048이나 다를 바 없음

	// 입력/출력 데이터 공간 할당
	A = (int*)malloc(datasize);
	B = (int*)malloc(datasize);
	C = (int*)malloc(datasize);

	// 입력 데이터의 초기화
	for (int i = 0; i < elements; i++) {
		A[i] = i;
		B[i] = i;
	}

	// 각 API 호출의 출력체크를 위해 사용함
	cl_int status;

	// 플랫폼의 수를 가져옴
	cl_uint numPlatforms = 0;
	status = clGetPlatformIDs(0, NULL, &numPlatforms);

	// 각 플랫폼을 위한 충분한 공간 할당
	cl_platform_id *platforms = NULL;
	platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));

	// 플랫폼 정보를 가져옴
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);

	// 디바이스 수를 가져옴
	cl_uint numDevices = 0;
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);

	// 각 디바이스를 위한 충분한 공간을 할당
	cl_device_id *devices;
	devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));

	// 디바이스 정보를 가져옴
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);

	// 컨텍스트를 생성하고 디바이스와 연결시킴**
	cl_context context;
	context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);

	// 명령어 큐를 생성하고 디바이스와 연결시킴
	cl_command_queue cmdQueue;
	cmdQueue = clCreateCommandQueue(context, devices[0], 0, &status);

	// 호스트 배열 A로부터 데이터를 포함하는 버퍼 객체 생성
	cl_mem bufA;
	bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);

	// 호스트 배열 B로부터 데이터를 포함하는 버퍼 객체 생성
	cl_mem bufB;
	bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);

	// 출력 데이터를 저장할 버퍼 객체를 생성
	cl_mem bufC;
	bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize, NULL, &status);

	// 입력 배열 A를 디바이스 버퍼 A에 작성
	status = clEnqueueWriteBuffer(cmdQueue, bufA, CL_FALSE, 0, datasize, A, 0, NULL, NULL);

	// 입력 배열 B를 디바이스 버퍼 B에 작성
	status = clEnqueueWriteBuffer(cmdQueue, bufB, CL_FALSE, 0, datasize, B, 0, NULL, NULL);

	// 소스코드를 갖고 있는 프로그램을 생성
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&programSource, (const size_t*)&source_size, &status);

	// 디바이스를 위한 프로그램 빌드(컴파일)
	status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);

	// 벡터 덧셈 커널을 생성
	cl_kernel kernel;
	kernel = clCreateKernel(program, "vecadd", &status);

	// 입력 및 출력 버퍼와 커널을 연결
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);

	// 실행을  위해 워크아이템의 인덱스 공간 (글로벌 워크사이즈)를 정의함
	// 워크그룹 크기(로컬 워크 사이즈)가 필요하지는 않지만 사용될 수 있다.
	size_t globalWorkSize[1];

	// 워크아이템의 '항목'이 있다.
	globalWorkSize[0] = elements;

	// 실행을 위해 커널을 실행함
	status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

	// 디바이스 출력 버퍼에서 호스트 출력 버퍼로 읽어옴
	clEnqueueReadBuffer(cmdQueue, bufC, CL_TRUE, 0, datasize, C, 0, NULL, NULL);

	// 출력 검증
	int result = 1;
	for (int i = 0; i < elements; i++) {
		printf("C[%d] : %d\n", i, C[i]);
		if (C[i] != i + i) {
			result = 0;
			break;
		}
	}
	if (result) {
		printf("Output is correct\n");
	}
	else {
		printf("Output is incorrect\n");
	}

	// OpenCl 리소스 해제
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(bufA);
	clReleaseMemObject(bufB);
	clReleaseMemObject(bufC);
	clReleaseContext(context);

	// 호스트 리소스 해제
	free(A);
	free(B);
	free(C);
	free(platforms);
	free(devices);

	return EXIT_SUCCESS;
}

