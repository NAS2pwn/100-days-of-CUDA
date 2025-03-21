#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"

#define KERNEL_SIZE 40
#define KERNEL_RADIUS (KERNEL_SIZE / 2)
#define SIGMA 5.0f

#define BLOCK_SIZE 16

void createGaussianKernelWithoutNormalization(float *kernel) {
    for (int y = 0; y < KERNEL_SIZE; y++) {
        for (int x = 0; x < KERNEL_SIZE; x++) {
            int dx = x - KERNEL_RADIUS;
            int dy = y - KERNEL_RADIUS;
            
            float value = expf(-(dx*dx + dy*dy) / (2 * SIGMA * SIGMA));
            
            kernel[y * KERNEL_SIZE + x] = value;
        }
    }
}

void createGaussianKernelWithNormalization(float *kernel) {
    float sum = 0.0f;

    for (int y = 0; y < KERNEL_SIZE; y++) {
        for (int x = 0; x < KERNEL_SIZE; x++) {
            int dx = x - KERNEL_RADIUS;
            int dy = y - KERNEL_RADIUS;
            
            float value = expf(-(dx*dx + dy*dy) / (2 * SIGMA * SIGMA));
            
            kernel[y * KERNEL_SIZE + x] = value;
            sum += value;
        }
    }

    for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; i++) {
        kernel[i] /= sum;
    }
}

__global__ void imageConvolutionKernel(unsigned char *input, unsigned char *output, float *kernel, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = threadIdx.z;

    if (x >= width || y >= height || c >= channels) return;

    float sum = 0.0f;

    for (int ky = -KERNEL_RADIUS; ky <= KERNEL_RADIUS; ky++) {
        for (int kx = -KERNEL_RADIUS; kx <= KERNEL_RADIUS; kx++) {
            int ix = x + kx;
            int iy = y + ky;

            ix = max(0, min(width - 1, ix));
            iy = max(0, min(height - 1, iy));

            int pixel_idx = (iy * width + ix) * channels + c;
            int kernel_idx = (ky + KERNEL_RADIUS) * KERNEL_SIZE + (kx + KERNEL_RADIUS);
            
            sum += kernel[kernel_idx] * input[pixel_idx];
        }
    }

    output[(y * width + x) * channels + c] = (unsigned char)min(255.0f, max(0.0f, sum));
}

int main(int argc, char **argv) {
    if (argc != 3 && argc != 4) {
        printf("Usage: %s <input_image> <output_image> [no_normalization]\n", argv[0]);
        return 1;
    }

    int width, height, channels;
    unsigned char* input_image = stbi_load(argv[1], &width, &height, &channels, 0);
    if (!input_image) {
        printf("Error loading image\n");
        return -1;
    }
    printf("Image loaded : %dx%d with %d channels\n", width, height, channels);

    unsigned char* output_image = (unsigned char*)malloc(width * height * channels);
    
    float* gaussian_kernel = (float*)malloc(KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    if (argc == 4) {
        printf("Using non-normalized Gaussian kernel\n");
        createGaussianKernelWithoutNormalization(gaussian_kernel);
    } else {
        printf("Using normalized Gaussian kernel\n");
        createGaussianKernelWithNormalization(gaussian_kernel);
    }

    unsigned char *d_input, *d_output;
    float *d_kernel;
    cudaMalloc(&d_input, width * height * channels * sizeof(unsigned char));
    cudaMalloc(&d_output, width * height * channels * sizeof(unsigned char));
    cudaMalloc(&d_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));

    cudaMemcpy(d_input, input_image, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, gaussian_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE, channels);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);

    printf("Launching kernel with grid: %dx%d, block: %dx%d\n", grid.x, grid.y, block.x, block.y);

    imageConvolutionKernel<<<grid, block>>>(d_input, d_output, d_kernel, width, height, channels);

    cudaMemcpy(output_image, d_output, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    stbi_write_jpg(argv[2], width, height, channels, output_image, 100);

    stbi_image_free(input_image);
    free(output_image);
    free(gaussian_kernel);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    return 0;
}

