__kernel void img_rotate(__global float* dest_data, __global float* src_data,
						 int W, int H,	// 이미지 크기 W for width, H for height
						 float sinTheta, float cosTheta){	// 회전 각도에 대한 인자
	// 워크아이템은 인덱스 공간(index space) 에서 인덱스(index)를 얻는다.
	const int ix = get_global_id(0);
	const int iy = get_global_id(1);

	// 데이터가 이동할 좌표 (ix, iy)를 계산한다.
	// 기준 좌표 x0, y0는 이미지의 중점이다.
	float x0 = W/2.0f;
	float y0 = W/2.0f;

	float x_diff = ix - x0;
	float y_diff = iy - y0;
	int xpos = (int)(x_diff*cosTheta + y_diff*sinTheta + x0);	// 평행이동후 회전변환 후 다시 평행이동을 나타냄.
	int ypos = (int)(y_diff*cosTheta - x_diff*sinTheta + y0);

	if(((int)xpos>=0) && ((int)xpos<W) && ((int)ypos>=0) && ((int)ypos<H)){	// 경계안에 있는 유효한 값에 대해서만
		// src_data의 (ix, iy) 값을 읽어 dest_data의 (xpos, ypos)에 저장한다.
		// 원점에 대해서 회전했기 때문에 변환할 필요는 없다.
		dest_data[iy*W+ix] = src_data[ypos*W+xpos];		// 이 식에 대해선 그림을 그려가면서 해보면 이해가 잘 된다.
														//	Width만 가지고 계산할 수 있는 것을 이해하는 것이 중요!
	}else{
		dest_data[iy*W+ix] = 0;		// NULL 값 찍히는 거 방지
	}
}