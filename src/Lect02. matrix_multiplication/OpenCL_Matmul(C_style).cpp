#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_SOURCE_SIZE (0x1000)
/*
	( 0  0  0 ) ---> rows (width)
	( 0  0  0 )
	( 0  0  0 )
	  |
	  |
	  v
	columns	(height)
									width, height 헷갈리면 안됨.
*/
int main()
{
	int i, j;
	int factor = 3;
	int mul_size = 16;
	int widthA = factor * mul_size;
	int heightA = factor * mul_size;
	int widthB = factor * mul_size;
	int heightB = factor * mul_size;
	int heightC = heightA, widthC = widthB;

	// A, B, C 행렬 메모리 할당
	float **A = (float**)malloc(widthA * sizeof(float*));
	for (i = 0; i < widthA; i++) A[i] = (float*)malloc(heightA * sizeof(float));
	float **B = (float**)malloc(widthB * sizeof(float*));
	for (i = 0; i < widthB; i++) B[i] = (float*)malloc(heightB * sizeof(float));
	float **C = (float**)malloc(widthC * sizeof(float*));
	for (i = 0; i < widthC; i++) B[i] = (float*)malloc(heightB * sizeof(float));

	// A, B, C 행렬 초기화
	for (i = 0; i < heightA; i++) {
		for (j = 0; j < widthA; j++) {
			A[i][j] = i + j;
		}
	}

	for (i = 0; i < heightB; i++) {
		for (j = 0; j < widthB; j++) {
			B[i][j] = i + j;
		}
	}

	cl_int err;
	try {

		// 플랫폼의 수를 가져옴
		cl_uint numPlatforms = 0;
		err = clGetPlatformIDs(0, NULL, &numPlatforms);

		// 각 플랫폼을 위한 충분한 공간 할당
		cl_platform_id *platforms = NULL;
		platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));

		// 플랫폼 정보를 가져옴
		err = clGetPlatformIDs(numPlatforms, platforms, NULL);

		// 디바이스 수를 가져옴
		cl_uint numDevices = 0;
		err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);

		// 각 디바이스를 위한 충분한 공간을 할당
		cl_device_id *devices;
		devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));

		// 디바이스 정보를 가져옴
		err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);

		// 컨텍스트를 생성하고 디바이스와 연결시킴**
		cl_context context;
		context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &err);

		// 명령어 큐를 생성하고 디바이스와 연결시킴
		cl_command_queue myqueue;
		myqueue = clCreateCommandQueue(context, devices[0], 0, &err);


		//// 첫 번째 플랫폼을 사용한다.
		//cl_platform_id platform;
		//err = clGetPlatformIDs(1, &platform, NULL);

		//// 첫 번째 디바이스를 사용한다.
		//cl_device_id device;
		//err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

		//cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };

		//// 컨텍스트를 생성한다.
		//cl_context context = clCreateContext(cps, 0, &device, NULL, NULL, &err);

		//// 명령어 큐를 생성한다.
		//cl_command_queue myqueue = clCreateCommandQueue(context, device, 0, &err);

		cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, widthA*heightA * sizeof(float), NULL, &err);
		cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, widthB*heightB * sizeof(float), NULL, &err);
		cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, widthC*heightC * sizeof(float), NULL, &err);

		err = clEnqueueWriteBuffer(myqueue, bufferA, CL_FALSE, 0, widthA*heightA * sizeof(float), A, 0, NULL, NULL);
		err = clEnqueueWriteBuffer(myqueue, bufferB, CL_FALSE, 0, widthB*heightB * sizeof(float), B, 0, NULL, NULL);

		// kernel source를 불러온다.
		FILE *fp;
		char fileName[] = "matmul_kernel.cl";
		/* 커널함수를 불러서 읽어온다. */
		fp = fopen(fileName, "r");
		if (!fp) {
			fprintf(stderr, "Failed to load kernel function.\n");
			exit(1);
		}
		/* 정상적으로 불러온 커널함수에 대하여 메모리 할당 및 size 측정 */
		char *programSource = (char*)malloc(MAX_SOURCE_SIZE);
		size_t source_size = fread(programSource, 1, MAX_SOURCE_SIZE, fp);
		fclose(fp);

		// 프로그램 소스 로드
		cl_program myprog = clCreateProgramWithSource(context, 1, (const char**)&programSource, (const size_t*)&source_size, &err);

		// 프로그램을 컴파일한다.
		// 컨텍스트에 있는 모든 디바이스에 대해 처리하도록 'device_list'에 NULL을 넘긴다.
		err = clBuildProgram(myprog, numDevices, devices, NULL, NULL, NULL);

		// 커널을 생성한다.
		cl_kernel mykernel = clCreateKernel(myprog, "matmul", &err);

		// 커널 인자를 설정한다.
		clSetKernelArg(mykernel, 0, sizeof(cl_mem), &bufferC);
		clSetKernelArg(mykernel, 1, sizeof(cl_int), &widthA);
		clSetKernelArg(mykernel, 2, sizeof(cl_int), &heightA);
		clSetKernelArg(mykernel, 3, sizeof(cl_int), &widthB);
		clSetKernelArg(mykernel, 4, sizeof(cl_int), &heightB);
		clSetKernelArg(mykernel, 5, sizeof(cl_mem), &bufferA);
		clSetKernelArg(mykernel, 6, sizeof(cl_mem), &bufferB);

		// 지역 / 전역 워크그룹 크기를 설정한다.
		// 행렬 크기가 16으로 나누어 떨어진다고 가정한다.
		size_t localws[2] = { 16, 16 };
		size_t globalws[2] = { widthC, heightC };

		// 커널을 실행한다.
		err = clEnqueueNDRangeKernel(myqueue, mykernel, 2, NULL, globalws, localws, 0, NULL, NULL);

		err = clEnqueueReadBuffer(myqueue, bufferC, CL_TRUE, 0, widthC*heightC * sizeof(float), C, 0, NULL, NULL);

		// print result
		for (i = 0; i < heightC; i++) {
			for (j = 0; j < widthC; j++) {
				printf("%lf ", C[i][j]);
			}
			printf("\n");
		}
	}
	catch(cl_int err){
		printf("Error : %d", err);
	}
    return EXIT_SUCCESS;
}

