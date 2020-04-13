__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

// When filter size 5x5, kernel data size = 5 x 5 x 32 = 800 bit  = 100 Byte < 1024 Bytes(maximum constant argument size)
// Using __constant data-type is very helpful

__kernel void gaussian_buffer_32FC1(
	__global uchar* src,
	__global float* dst,
	__constant float* gKernel,
	const int width,
	const int mSize)
{
	int col = get_global_id(0); // 0 ~ width - 1
	int row = get_global_id(1); // 0 ~ height - 1

	float sum = 0.f;

	int kSize = mSize * 2 + 1;
	for (int i = -mSize; i <= mSize; i++) { // row slice
		for (int j = -mSize; j <= mSize; j++) { // col slice
			sum += (float)(src[(row + i) * width + (col + j)]) * gKernel[(i + mSize) * (kSize) + (j + mSize)];
		}
	}
	dst[row * width + col] = sum;
}

__kernel void gaussian_img_32FC1(
	__read_only image2d_t src,	// image format : INTENSITY(1 channel), CL_UNSIGNED_INT8(uchar)
	__write_only image2d_t dst, // image format : INTENSITY(1 channel), CL_FLOAT(float)
	__constant float* gKernel,
	const int mSize) // mSize = kernel_size / 2
{
	int col = get_global_id(0); // 0 ~ width - 1
	int row = get_global_id(1); // 0 ~ height - 1

	const int2 pos = { col, row };

	float sum = 0.f;

	int kSize = mSize * 2 + 1;
	for (int i = -mSize; i <= mSize; i++) { // row slice
		for (int j = -mSize; j <= mSize; j++) { // col slice
			sum += (float)(read_imageui(src, sampler, pos + (int2)(j, i)).x) * gKernel[(i + mSize) * (kSize) + (j + mSize)];
			// read_imageui -> read image object created with image_channel_data_type : CL_UNSIGNED_INT8, CL_UNSIGNED_INT16, CL_UNSIGNED_INT32
		}
	}
	write_imagef(dst, pos, (float4)(sum)); // write image --- float4, INTENSITY
	// write_imagef
	// Channel type
	// 1. CL_R ----> float4(r, 0, 0, 1)
	// 2. CL_A ----> float4(0, 0, 0, a)
	// 3. CL_RG ----> float4(r, g, 0, 1)
	// 4. CL_RA ----> float4(r, 0, 0, a)
	// 5. CL_RGB ---> float4(r, g, b, 1)
	// 6. CL_RGBA, CL_BGRA, CL_ARGB ----> float4(r, g, b, a)
	// 7. CL_INTENSITY ---> float4(I, I, I, I)
}