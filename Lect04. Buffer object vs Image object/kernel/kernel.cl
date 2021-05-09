__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

// When filter size 5x5, kernel data size = 5 x 5 x 32 = 800 bit  = 100 Byte <
// 1024 Bytes(maximum constant argument size) Using __constant data-type is very useful

__kernel void GaussianBuffer(
    __global uchar *src,
    __global uchar *dst,
    __constant float *gKernel,
    const int width,
    const int channels,
    const int mSize)
{
    int col = get_global_id(0); // 0 ~ width - 1
    int row = get_global_id(1); // 0 ~ height - 1

    float sumRGB[3] = {0.f};

    int kSize = mSize * 2 + 1;
    for (int i = -mSize; i <= mSize; i++) // row slice
    {
        for (int j = -mSize; j <= mSize; j++) // col slice
        {
            for (int ch = 0; ch < channels; ch++) // channel RGB or GRAY
            {
                sumRGB[ch] += convert_float(src[channels * ((row + i) * width + (col + j)) + ch]) *
                              gKernel[(i + mSize) * (kSize) + (j + mSize)];
            }
        }
    }

    for (int ch = 0; ch < channels; ch++) // channel RGB or GRAY
    {
        dst[channels * (row * width + col) + ch] = convert_uchar(sumRGB[ch]);
    }
}

/*
	write image --- float4, INTENSITY
	Channel type
	1. CL_R ----> float4(r, 0, 0, 1)
	2. CL_A ----> float4(0, 0, 0, a)
	3. CL_RG ----> float4(r, g, 0, 1)
	4. CL_RA ----> float4(r, 0, 0, a)
	5. CL_RGB ---> float4(r, g, b, 1)
	6. CL_RGBA, CL_BGRA, CL_ARGB ----> float4(r, g, b, a)
	7. CL_INTENSITY ---> float4(I, I, I, I)
*/

__kernel void GaussianImage(
    __read_only image2d_t src,  // image format : CL_RGBA(4 channel), CL_UNSIGNED_INT8(uchar)
    __write_only image2d_t dst, // image format : CL_RGBA(4 channel), CL_UNSIGNED_INT8(uchar)
    __constant float *gKernel,
    const int mSize) // mSize = kernel_size / 2
{
    int col = get_global_id(0); // 0 ~ width - 1
    int row = get_global_id(1); // 0 ~ height - 1

    int2 pos = {col, row};

    float4 sumRGB = 0.f;

    int kSize = mSize * 2 + 1;
    for (int i = -mSize; i <= mSize; i++) // row slice
    {
        for (int j = -mSize; j <= mSize; j++) // col slice
        {
            sumRGB += convert_float4(read_imageui(src, sampler, pos + (int2)(j, i))) *
                      gKernel[(i + mSize) * (kSize) + (j + mSize)];
            // read_imageui -> read image object created with image_channel_data_type
            // : CL_UNSIGNED_INT8, CL_UNSIGNED_INT16, CL_UNSIGNED_INT32
        }
    }
    write_imageui(dst, pos, convert_uint4(sumRGB));
}