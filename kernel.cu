#include "kernel.h"
#include <stdio.h>
#include <stdint.h>

// Calculate the number of iterations required for a complex number to escape the Mandelbrot set
// Based on: http://selkie.macalester.edu/csinparallel/modules/CUDAArchitecture/build/html/1-Mandelbrot/Mandelbrot.html
__device__
uint32_t mandel_double(double cr, double ci, int max_iter) {
    double zr = 0;      // real part of z
    double zi = 0;      // imaginary part of z
    double zrsqr = 0;   // real part of z squared
    double zisqr = 0;   // imaginary part of z squared

    /*
     * The Mandelbrot set is the set of complex numbers c for which the sequence z(n+1) = z(n)^2 + c does not diverge.
     * The sequence is considered to diverge if the magnitude of z(n) exceeds 2.
     * The magnitude of a complex number z = a + bi is given by |z| = sqrt(a^2 + b^2).
    */

    // If the number of iterations exceeds the maximum number of iterations, the complex number is considered to be in the Mandelbrot set
    uint32_t iter = 0;

    for (iter = 0; iter < max_iter; ++iter) {   // Iteration loop, up to a maximum number of iterations
        zi = zr * zi;               // Multiply the real part (zr) and the imaginary part (zi) of the complex number
        zi += zi;                   // Double the imaginary part
        zi += ci;                   // Add the constant imaginary part (ci)
        zr = zrsqr - zisqr + cr;    // Calculate the new real part as the difference of the squares of the real and imaginary parts, plus the constant real part (cr)
        zrsqr = zr * zr;            // Square the real part
        zisqr = zi * zi;            // Square the imaginary part

        if (zrsqr + zisqr > 4.0) { // If the sum of the squares of the real and imaginary parts is greater than 4, the number is not in the Mandelbrot set
            break; // Break the loop
        }
    }

    return iter;
}

__device__
// Map pixel coordinates to mandelbrot set coordinates
void map_pixel_to_mandelbrot(
                            int x, int y,               // Pixel coordinates
                            int width, int height,      // Image dimensions
                            double x_min, double x_max, // Mandelbrot set coordinates
                            double y_min, double y_max, // Mandelbrot set coordinates
                            double *cr, double *ci      // Output: Mandelbrot set coordinates
                            )
{
    *cr = x * (x_max - x_min) / width + x_min;
    *ci = y * (y_max - y_min) / height + y_min;
}

__global__
// Kernel function to calculate the Mandelbrot set
void mandelbrot_kernel(
                        uint32_t *output,           // Output: number of iterations required for a complex number to escape the Mandelbrot set
                        int width, int height,      // Image dimensions
                        double x_min, double x_max, // Mandelbrot set coordinates
                        double y_min, double y_max, // Mandelbrot set coordinates
                        int max_iter                // Maximum number of iterations
                        )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the x coordinate of the pixel
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Calculate the y coordinate of the pixel

    if (x < width && y < height) { // Check if the pixel is within the image dimensions
        double cr, ci; // Mandelbrot set coordinates

        // Map pixel coordinates to mandelbrot set coordinates
        map_pixel_to_mandelbrot(x, y, width, height, x_min, x_max, y_min, y_max, &cr, &ci);

        // Calculate the number of iterations required for a complex number to escape the Mandelbrot set
        output[y * width + x] = mandel_double(cr, ci, max_iter);
    }
}

void mandelbrot(
                uint32_t *output,           // Output: number of iterations required for a complex number to escape the Mandelbrot set
                int width, int height,      // Image dimensions
                double x_min, double x_max, // Mandelbrot set coordinates
                double y_min, double y_max, // Mandelbrot set coordinates
                int max_iter
                )
{
    /*
     * The block dimensions are chosen to be 16x16, which is a common choice for the dimensions of a block.
     * The grid dimensions are calculated based on the image dimensions and the block dimensions.
    */
    dim3 block(16, 16); // Block dimensions
    /*
     * Grid dimensions are calculated as follows:
     * 1. The number of blocks in the x direction is the width of the image divided by the x dimension of the block, rounded up to the nearest integer.
     * 2. The number of blocks in the y direction is the height of the image divided by the y dimension of the block, rounded up to the nearest integer.
     * This ensures that the grid covers the entire image, even if the image dimensions are not a multiple of the block dimensions.
    */
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y); // Grid dimensions

    mandelbrot_kernel<<<grid, block>>>(output, width, height, x_min, x_max, y_min, y_max, max_iter); // Launch the kernel
}