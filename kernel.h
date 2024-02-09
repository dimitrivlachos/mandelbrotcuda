#pragma once
#include <stdint.h>

struct uchar4;

void mandelbrot(
                uint32_t *output,           // Output: number of iterations required for a complex number to escape the Mandelbrot set
                int width, int height,      // Image dimensions
                double x_min, double x_max, // Mandelbrot set coordinates
                double y_min, double y_max, // Mandelbrot set coordinates
                int max_iter                // Maximum number of iterations
                );