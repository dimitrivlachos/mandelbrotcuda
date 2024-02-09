#include "kernel.h"
#include <stdio.h>
#include <stdint.h>

__device__
// Calculate the number of iterations required for a complex number to escape the Mandelbrot set
uint32_t mandel_double(double cr, double ci, int max_iter) {
    double zr = 0;      // real part of z
    double zi = 0;      // imaginary part of z
    double zrsqr = 0;   // real part of z squared
    double zisqr = 0;   // imaginary part of z squared

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

__global__
// Kernel function to calculate the Mandelbrot set
void mandel_kernel(uint32_t *counts, double xmin, double ymin, double step, int max_iter, int dim, uint32_t *colors) {
    int pix_per_thread = dim * dim / (gridDim.x * blockDim.x); // Calculate the number of pixels per thread
    int tId = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the thread ID
    int offset = tId * pix_per_thread; // Calculate the offset for the thread

    for (int i = offset; i < offset + pix_per_thread; ++i) { // Loop over the pixels
        int x = i % dim; // Calculate the x coordinate of the pixel
        int y = i / dim; // Calculate the y coordinate of the pixel

        double cr = xmin + x * step; // Calculate the real part of the complex number
        double ci = ymin + y * step; // Calculate the imaginary part of the complex number

        counts[y * dim + x]  = colors[mandel_double(cr, ci, max_iter)]; // Calculate the number of iterations required for the complex number to escape the Mandelbrot set
    }
    if (gridDim.x * blockDim.x * pix_per_thread < dim * dim && tId < (dim * dim) - (blockDim.x * gridDim.x)) {
        int i = gridDim.x * blockDim.x * pix_per_thread + tId;
        int x = i % dim; // Calculate the x coordinate of the pixel
        int y = i / dim; // Calculate the y coordinate of the pixel

        double cr = xmin + x * step; // Calculate the real part of the complex number
        double ci = ymin + y * step; // Calculate the imaginary part of the complex number

        counts[y * dim + x]  = colors[mandel_double(cr, ci, max_iter)]; // Calculate the number of iterations required for the complex number to escape the Mandelbrot set
    }
}

// Function to calculate the Mandelbrot set
void mandelbrot(uint32_t *counts, double xmin, double ymin, double step, int max_iter, int dim, uint32_t *colors) {
    dim3 block(32, 32); // Declare a block of 32x32 threads
    dim3 grid(dim / block.x, dim / block.y); // Declare a grid of blocks

    uint32_t *d_counts; // Declare a pointer to the device memory for the counts
    uint32_t *d_colors; // Declare a pointer to the device memory for the colors

    cudaMalloc((void **)&d_counts, dim * dim * sizeof(uint32_t)); // Allocate memory on the device for the counts
    cudaMalloc((void **)&d_colors, max_iter * sizeof(uint32_t)); // Allocate memory on the device for the colors

    cudaMemcpy(d_colors, colors, max_iter * sizeof(uint32_t), cudaMemcpyHostToDevice); // Copy the colors to the device

    mandel_kernel<<<grid, block>>>(d_counts, xmin, ymin, step, max_iter, dim, d_colors); // Call the kernel function

    cudaMemcpy(counts, d_counts, dim * dim * sizeof(uint32_t), cudaMemcpyDeviceToHost); // Copy the counts from the device

    cudaFree(d_counts); // Free the device memory for the counts
    cudaFree(d_colors); // Free the device memory for the colors
}