__kernel void matmul(__global float* outputC, int widthA, int heightA, int widthB, int heightB, __global float* inputA, __global float* inputB){
	// Y ������ ���� ��ġ�� �����´�.
	int row = get_global_id(1);
	// X ������ ���� ��ġ�� �����´�.
	int col = get_global_id(0);

	float sum = 0.0f;

	// ��� C�� �� ��ҿ� ���� ��� ���� ����Ѵ�.
	for(int i=0; i< widthA; i++){
		sum += inputA[row*widthA+i]*inputB[i*widthB+col];
	}

	outputC[row*widthB+col]=sum;
}