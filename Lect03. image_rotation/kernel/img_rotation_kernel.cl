__kernel void image_rotate(
    __global uchar *dest_data,
    __global uchar *src_data,
    int width,
    int height,
    float sin_theta,
    float cos_theta)
{
    size_t col = get_global_id(0);
    size_t row = get_global_id(1);

    float cx = (float)width / 2.0f;
    float cy = (float)height / 2.0f;
    float x_diff = col - cx;
    float y_diff = row - cy;

    int xpos = convert_int(x_diff * cos_theta + y_diff * sin_theta + cx);
    int ypos = convert_int(y_diff * cos_theta - x_diff * sin_theta + cy);
    
    if ((xpos >= 0) && (xpos < width) && (ypos >= 0) && (ypos < height))
    {
        for(int ch = 0; ch < 3; ch++){
            dest_data[3 * (row * width + col) + ch] = src_data[3 * (ypos * width + xpos) + ch];
        }
    }
    else
    {
        dest_data[3 * (row * width + col) + 0] = 0; // R
        dest_data[3 * (row * width + col) + 1] = 0; // G
        dest_data[3 * (row * width + col) + 2] = 0; // B
    }
}