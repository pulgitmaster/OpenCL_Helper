#include "ocl.hpp"
// bitmap class reference: https://github.com/ArashPartow/bitmap
#include "bitmap.hpp"

#ifndef uchar
typedef unsigned char uchar;
#endif

#ifndef PI
#define PI 3.14159265359
#endif

int main(int argc, char *argv[])
{
    OCL ocl;
    if (ocl.Initialized() == false)
    {
        std::cerr << "ocl.Initialized() is failed\n";
        return -1;
    }

    int device_num = 0;
    ocl.GetDeviceInfo(device_num);
    if (ocl.BuildProgram("../kernel/img_rotation_kernel.cl", device_num) < 0)
    {
        return -1;
    }

    // Generate kernel
    cl::Kernel kernel_image_rotate;
    if (ocl.CreateKernel(kernel_image_rotate, "image_rotate") < 0)
    {
        return -1;
    }

    // Read input image
    Bitmap image("../input.bmp");
    // Create output image
    Bitmap out_image(image.width(), image.height());

    if (!image)
    {
        std::cerr << "Error - failed to open: input.bmp\n";
        return -1;
    }

    double degree = 60; // Roate image 60' clock_wise
    double theta = (double)(PI / (double)(180 / degree));
    float cos_theta = static_cast<float>(std::cos(theta));
    float sin_theta = static_cast<float>(std::sin(theta));

	cl_int err = CL_SUCCESS;
    cl::Event event;

    // Input Buffer and output Buffer
    size_t data_size = 3 * image.width() * image.height();
    cl::Buffer d_ip = cl::Buffer(ocl.context, CL_MEM_READ_ONLY, data_size * sizeof(uchar));
    cl::Buffer d_op = cl::Buffer(ocl.context, CL_MEM_WRITE_ONLY, data_size * sizeof(uchar));
    ocl.cmdQueue.enqueueWriteBuffer(d_ip, CL_TRUE, 0, data_size * sizeof(uchar), image.data());

    // Set kernel arguments
    kernel_image_rotate.setArg(0, d_op);
    kernel_image_rotate.setArg(1, d_ip);
    kernel_image_rotate.setArg(2, image.width());
    kernel_image_rotate.setArg(3, image.height());
    kernel_image_rotate.setArg(4, sin_theta);
    kernel_image_rotate.setArg(5, cos_theta);

    // Launch kernel
    ocl.cmdQueue.enqueueNDRangeKernel(kernel_image_rotate, cl::NullRange, cl::NDRange(image.width(), image.height()), cl::NullRange, NULL, &event);
    event.wait();

    // Read from device to host
    ocl.cmdQueue.enqueueReadBuffer(d_op, CL_TRUE, 0, data_size * sizeof(uchar), out_image.data());

    // Save output image
    out_image.save_image("output.bmp");

    return 0;
}