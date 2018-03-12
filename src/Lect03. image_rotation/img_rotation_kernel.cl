__kernel void img_rotate(__global float* dest_data, __global float* src_data,
						 int W, int H,	// �̹��� ũ�� W for width, H for height
						 float sinTheta, float cosTheta){	// ȸ�� ������ ���� ����
	// ��ũ�������� �ε��� ����(index space) ���� �ε���(index)�� ��´�.
	const int ix = get_global_id(0);
	const int iy = get_global_id(1);

	// �����Ͱ� �̵��� ��ǥ (ix, iy)�� ����Ѵ�.
	// ���� ��ǥ x0, y0�� �̹����� �����̴�.
	float x0 = W/2.0f;
	float y0 = W/2.0f;

	float x_diff = ix - x0;
	float y_diff = iy - y0;
	int xpos = (int)(x_diff*cosTheta + y_diff*sinTheta + x0);	// �����̵��� ȸ����ȯ �� �ٽ� �����̵��� ��Ÿ��.
	int ypos = (int)(y_diff*cosTheta - x_diff*sinTheta + y0);

	if(((int)xpos>=0) && ((int)xpos<W) && ((int)ypos>=0) && ((int)ypos<H)){	// ���ȿ� �ִ� ��ȿ�� ���� ���ؼ���
		// src_data�� (ix, iy) ���� �о� dest_data�� (xpos, ypos)�� �����Ѵ�.
		// ������ ���ؼ� ȸ���߱� ������ ��ȯ�� �ʿ�� ����.
		dest_data[iy*W+ix] = src_data[ypos*W+xpos];		// �� �Ŀ� ���ؼ� �׸��� �׷����鼭 �غ��� ���ذ� �� �ȴ�.
														//	Width�� ������ ����� �� �ִ� ���� �����ϴ� ���� �߿�!
	}else{
		dest_data[iy*W+ix] = 0;		// NULL �� ������ �� ����
	}
}