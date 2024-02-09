#include "kernel.h"
#define cimg_display 0 // Disable CImg display
#include "dependencies/CImg/CImg.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <string>

int main(int argc, char* argv[]) {
    // Check for command line arguments
    if (argc != 2) {
        printf("Usage: ./mandelbrot <size>\n");
        return 1;
    }

    // Get size from command line arguments
    int size = std::stoi(argv[1]);

    uint32_t *d_out;
    cudaMalloc(&d_out, size * size * sizeof(uint32_t));

    // Call mandelbrot kernel
    mandelbrot(d_out, size, size, -2.0, 1.0, -1.0, 1.0, 1000);
    
    // Copy output from device to host
    uint32_t *out = (uint32_t *)malloc(size * size * sizeof(uint32_t));
    cudaMemcpy(out, d_out, size * size * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Initialize output image
    cimg_library::CImg<unsigned char> outImage(size, size, 1, 3, 0);

    // Set output image data
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            outImage(i, j, 0, 0) = (out[i + j * size] >> 16) & 0xFF;
            outImage(i, j, 0, 1) = (out[i + j * size] >> 8) & 0xFF;
            outImage(i, j, 0, 2) = out[i + j * size] & 0xFF;
        }
    }

    // Save output image
    outImage.save("mandelbrot.bmp");

    // Free device memory
    cudaFree(d_out);

    return 0;
}