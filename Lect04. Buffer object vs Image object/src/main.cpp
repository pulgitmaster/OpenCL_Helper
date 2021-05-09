#include "ocl.hpp"
// bitmap class reference: https://github.com/ArashPartow/bitmap
#include "bitmap.hpp"

#ifndef uchar
typedef unsigned char uchar;
#endif

#ifndef PI
#define PI 3.14159265359
#endif

void GaussianBufferObject(const OCL &ocl, cl::Kernel &kernel, const Bitmap &bitmapInput, Bitmap &bitmapOutput);
void GaussianImageObject(const OCL &ocl, cl::Kernel &kernel, const Bitmap &bitmapInput, Bitmap &bitmapOutput);
void RGB2RGBA(const uchar *rgb, uchar *rgba, unsigned int width, unsigned int height);
void RGBA2RGB(const uchar *rgba, uchar *rgb, unsigned int width, unsigned int height);
void FilterCreation1D(float *GKernel, int kSize, double sigma);
double ExecutionTime(cl::Event event);

int main(int argc, char *argv[])
{
    OCL ocl(0, "GPU", 0);
    if (ocl.Initialized() == false)
    {
        std::cerr << "ocl.Initialized() is failed\n";
        return -1;
    }

    int device_num = 0;
    ocl.GetDeviceInfo(device_num);
    if (ocl.BuildProgram("../kernel/kernel.cl", device_num) < 0)
    {
        return -1;
    }

    // Generate kernel
    cl::Kernel kernel_buffer;
    if (ocl.CreateKernel(kernel_buffer, "GaussianBuffer") < 0)
    {
        return -1;
    }

    cl::Kernel kernel_image;
    if (ocl.CreateKernel(kernel_image, "GaussianImage") < 0)
    {
        return -1;
    }

    // Read input image
    Bitmap bitmapInput("../input.bmp");
    // Create output image
    Bitmap bitmapOutputBuffer(bitmapInput.width(), bitmapInput.height());
    Bitmap bitmapOutputImage(bitmapInput.width(), bitmapInput.height());

    GaussianBufferObject(ocl, kernel_buffer, bitmapInput, bitmapOutputBuffer);
    bitmapOutputBuffer.save_image("gaussian_buffer.bmp");
    GaussianImageObject(ocl, kernel_image, bitmapInput, bitmapOutputImage);
    bitmapOutputImage.save_image("gaussian_image.bmp");

    return 0;
}

void GaussianBufferObject(const OCL &ocl, cl::Kernel &kernel, const Bitmap &bitmapInput, Bitmap &bitmapOutput)
{
    const unsigned int width = bitmapInput.width();
    const unsigned int height = bitmapInput.height();
    const int channels = 3;
    size_t data_size = channels * width * height;

    cl::Buffer dev_ImgSrc = cl::Buffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, data_size * sizeof(uchar), (uchar *)bitmapInput.data());
    cl::Buffer dev_ImgDst = cl::Buffer(ocl.context, CL_MEM_WRITE_ONLY, data_size * sizeof(uchar));

    // Create Gaussian filter mask
    int kSize = 5;
    int mSize = kSize / 2;
    float *gKernel = new float[kSize * kSize];
    FilterCreation1D(gKernel, kSize, 1.5);

    // Create buffer for mask and transfer it to the device
    cl::Buffer dev_gKernel = cl::Buffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * kSize * kSize, gKernel);

    // Run Gaussian kernel
    cl::Event gaussian_buffer_evt;
    kernel.setArg(0, dev_ImgSrc);
    kernel.setArg(1, dev_ImgDst);
    kernel.setArg(2, dev_gKernel);
    kernel.setArg(3, width);
    kernel.setArg(4, channels);
    kernel.setArg(5, mSize);
    ocl.cmdQueue.enqueueNDRangeKernel(kernel, cl::NDRange(mSize, mSize), cl::NDRange(width - 2 * mSize, height - 2 * mSize), cl::NullRange, NULL, &gaussian_buffer_evt);
    gaussian_buffer_evt.wait();

    // Transfer buffer back to host
    ocl.cmdQueue.enqueueReadBuffer(dev_ImgDst, CL_TRUE, 0, data_size * sizeof(uchar), bitmapOutput.data());
    fprintf(stdout, "Execution time(%s) : %.5lfms\n", "GaussianBuffer", ExecutionTime(gaussian_buffer_evt));

    // Release memory
    delete[] gKernel;
}

void GaussianImageObject(const OCL &ocl, cl::Kernel &kernel, const Bitmap &bitmapInput, Bitmap &bitmapOutput)
{
    const unsigned int width = bitmapInput.width();
    const unsigned int height = bitmapInput.height();
    const int channels = 4;
    size_t data_size = channels * width * height;
    cl_int err = CL_SUCCESS;

    uchar* rgba_input = new uchar[data_size];
    uchar* rgba_output = new uchar[data_size];
    RGB2RGBA(bitmapInput.data(), rgba_input, width, height);

    // Create an OpenCL Image / texture and transfer data to the device
    cl::Image2D dev_ImgSrc;
    try
    {
        dev_ImgSrc = cl::Image2D(ocl.context,
                                 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
                                 width, height,
                                 /*length of row*/ 0, //sizeof(uchar) * width,
                                 rgba_input, &err);
        // ref : https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/cl_image_format.html
    }
    catch (cl::Error &e)
    {
        std::cout << e.what() << "(" << e.err() << ")"
                  << "\n";
    }

    // Create a buffer for the result
    cl::Image2D dev_ImgDst = cl::Image2D(ocl.context,
                                         CL_MEM_WRITE_ONLY,
                                         cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
                                         width, height,
                                         0, ///*length of row*/ sizeof(uchar) * width,
                                         NULL, &err);

    // Create Gaussian filter mask
    int kSize = 5;
    float *gKernel = new float[kSize * kSize];
    FilterCreation1D(gKernel, kSize, 1.5);

    // Create buffer for mask and transfer it to the device
    cl::Buffer dev_gKernel = cl::Buffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * kSize * kSize, gKernel);

    // Run Gaussian kernel
    cl::Event gaussian_img_evt;
    kernel.setArg(0, dev_ImgSrc);
    kernel.setArg(1, dev_ImgDst);
    kernel.setArg(2, dev_gKernel);
    kernel.setArg(3, kSize / 2);
    ocl.cmdQueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(width, height), cl::NullRange, NULL, &gaussian_img_evt);
    gaussian_img_evt.wait();

    // Transfer image back to host
    cl::size_t<3> origin;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    cl::size_t<3> region;
    region[0] = width;
    region[1] = height;
    region[2] = 1;
    ocl.cmdQueue.enqueueReadImage(dev_ImgDst, CL_TRUE, origin, region, sizeof(uchar) * width * channels, 0, rgba_output);
    fprintf(stdout, "Execution time(%s) : %.5lfms\n", "GaussianImage", ExecutionTime(gaussian_img_evt));

    RGBA2RGB(rgba_output, bitmapOutput.data(), width, height);

    // Release memory
    delete[] rgba_input;
    delete[] rgba_output;
    delete[] gKernel;
}

// Function that convert RGB format to RGBA
void RGB2RGBA(const uchar *rgb, uchar *rgba, unsigned int width, unsigned int height)
{
    int cnt = width * height;
    for (int i = cnt; --i; rgba += 4, rgb += 3)
    {
        *(uint32_t *)(void *)rgba = *(const uint32_t *)(const void *)rgb;
    }
    for (int j = 0; j < 3; ++j)
    {
        rgba[j] = rgb[j];
    }
}

// Function that convert RGBA format to RGB
void RGBA2RGB(const uchar *rgba, uchar *rgb, unsigned int width, unsigned int height)
{
    int cnt = width * height;
    for (int i = cnt; --i; rgba += 4, rgb += 3)
    {
        memcpy((void *)rgb, (const void *)rgba, sizeof(uchar) * 3);
    }
    for (int j = 0; j < 3; ++j)
    {
        rgb[j] = rgba[j];
    }
}

// Function to create Gaussian filter
void FilterCreation1D(float *GKernel, int kSize, double sigma)
{
    // intialising standard deviation to 1.0
    double r, s = 2.0 * sigma * sigma;

    // sum is for normalization
    float sum = 0.f;

    int m_size = (kSize / 2);

    // generating kSzie x kSize kernel
    for (int x = -m_size; x <= m_size; x++)
    {
        for (int y = -m_size; y <= m_size; y++)
        {
            r = (x * x + y * y);
            GKernel[(x + m_size) * (kSize) + (y + m_size)] = static_cast<float>((exp((-r) / s)) / (PI * s));
            sum += GKernel[(x + m_size) * (kSize) + (y + m_size)];
        }
    }

    // normalizing the Kernel
    for (int i = 0; i < kSize; i++)
        for (int j = 0; j < kSize; j++)
            GKernel[i * kSize + j] /= sum;
}

double ExecutionTime(cl::Event event)
{
    cl_ulong start, end;

    start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

    return (double)1.0e-6 * (double)(end - start); // convert nanoseconds to milli-seconds on return
}