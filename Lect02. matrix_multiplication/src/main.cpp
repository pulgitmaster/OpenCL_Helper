#include "ocl.hpp"
#include "mat.hpp"

void OclMatMul(const OCL& ocl, cl::Kernel& kernel, Mat<float>& A, const Mat<float>& B, Mat<float>& C)
{
	// Initialize C with 0
	C.SetDataWithScalar(0.f);

	cl::Buffer bufferA = cl::Buffer(ocl.context, CL_MEM_READ_ONLY, A.GetWidth() * A.GetHeight() * sizeof(float));
	cl::Buffer bufferB = cl::Buffer(ocl.context, CL_MEM_READ_ONLY, B.GetWidth() * B.GetHeight() * sizeof(float));
	cl::Buffer bufferC = cl::Buffer(ocl.context, CL_MEM_WRITE_ONLY, C.GetWidth() * C.GetHeight() * sizeof(float));

	ocl.cmdQueue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, A.GetWidth() * A.GetHeight() * sizeof(float), A.GetData());
	ocl.cmdQueue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, B.GetWidth() * B.GetHeight() * sizeof(float), B.GetData());

	cl::Event event;
	kernel.setArg(0, bufferC);
	kernel.setArg(1, A.GetWidth());
	kernel.setArg(2, A.GetHeight());
	kernel.setArg(3, B.GetWidth());
	kernel.setArg(4, B.GetHeight());
	kernel.setArg(5, bufferA);
	kernel.setArg(6, bufferB);

	ocl.cmdQueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(C.GetWidth(), C.GetHeight()), cl::NullRange, NULL, &event);
	event.wait();

	ocl.cmdQueue.enqueueReadBuffer(bufferC, CL_TRUE, 0, C.GetWidth() * C.GetHeight() * sizeof(float), C.GetDataAddr(0));
}

void NativeMatMul(const Mat<float>& A, const Mat<float>& B, Mat<float>& C)
{
	// [m, k] x [k, n] = [m, n]
	int m = A.GetHeight();
	int k = A.GetWidth();
	int n = B.GetWidth();

	if (k != B.GetHeight())
	{
		std::cerr << "invalid multiplication between: " \
			<< "[" << A.GetHeight() << ", " << A.GetWidth() << "] * " \
			<< "[" << B.GetHeight() << ", " << B.GetWidth() << "]\n";
		return;
	}

	// Initialize C with 0
	C.SetDataWithScalar(0.f);
	for (int i1 = 0; i1 < m; i1++) {
		for (int i2 = 0; i2 <n; i2++) {
			for (int i3 = 0; i3 < k; i3++) {
				C[n*i1 + i2] += A[i1 * k + i3] * B[i3 * n + i2];
			}
		}
	}
}

int main(int argc, char *argv[])
{
    OCL ocl;
    if(ocl.Initialized() == false)
    {
        std::cerr << "ocl.Initialized() is failed\n";
        return -1;
    }

    int device_num = 0;
    ocl.GetDeviceInfo(device_num);
	if (ocl.BuildProgram("../kernel/matmul_kernel.cl", device_num) < 0)
	{
		return -1;
	}

	// Generate kernel
	cl::Kernel kernel_matmul;
	if (ocl.CreateKernel(kernel_matmul, "matmul") < 0)
	{
		return -1;
	}
    
	// Size of Matrix A : [3f x 4f], Size of Matrix B : [4f x 3f] (f means factor)
	const int factor = 4;
	const int heightA = 3 * factor;
	const int widthA = 4 * factor;
	const int heightB = 4 * factor;
	const int widthB = 3 * factor;
	int widthC = widthB, heightC = heightA;

	// Matrix A, B declaration
	float *A = new float[heightA * widthA];
	float *B = new float[heightB * widthB];

	// Matrix A, B initilization
	for (int i = 0; i < heightA * widthA; i++) {
		A[i] = (float)(i % 10) + 0.1f;
	}
	for (int i = 0; i < heightB * widthB; i++) {
		B[i] = (float)(i % 10) + 0.1f;
	}

	Mat<float> matA(widthA, heightA, A);
	Mat<float> matB(widthB, heightB, B);
	Mat<float> matC(widthC, heightC);

	NativeMatMul(matA, matB, matC);
	std::cout << "After NativeMatMul:\n";
	matC.Print();

	OclMatMul(ocl, kernel_matmul, matA, matB, matC);
	std::cout << "After OclMatMul:\n";
	matC.Print();

	// Release memory
	delete[] A;
	delete[] B;
    return 0;
}