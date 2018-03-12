__kernel void matmul(__global float* outputC, int widthA, int heightA, int widthB, int heightB, __global float* inputA, __global float* inputB){
	// Y 방향의 전역 위치를 가져온다.
	int row = get_global_id(1);
	// X 방향의 전역 위치를 가져온다.
	int col = get_global_id(0);

	float sum = 0.0f;

	// 행렬 C의 한 요소에 대한 결과 값을 계산한다.
	for(int i=0; i< widthA; i++){
		sum += inputA[row*widthA+i]*inputB[i*widthB+col];
	}

	outputC[row*widthB+col]=sum;
}