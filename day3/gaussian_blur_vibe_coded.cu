#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"

#define KERNEL_SIZE 30
#define KERNEL_RADIUS (KERNEL_SIZE / 2)

#define SIGMA 5.0f

#define BLOCK_SIZE 16

void createGaussianKernel(float *kernel, int size, float sigma) {
    int center = size / 2;
    float sum = 0.0f;
    
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int dx = x - center;
            int dy = y - center;
            
            // Formule gaussienne 2D
            float value = expf(-(dx*dx + dy*dy) / (2 * sigma * sigma));
            
            kernel[y * size + x] = value;
            sum += value;
        }
    }
    
    for (int i = 0; i < size * size; i++) {
        kernel[i] /= sum;
    }
    
    printf("Noyau gaussien (%dx%d):\n", size, size);
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            printf("%.4f ", kernel[y * size + x]);
        }
        printf("\n");
    }
}

__global__ void gaussianBlurKernel(unsigned char *input, unsigned char *output, 
                                  float *kernel, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        for (int ky = -KERNEL_RADIUS; ky <= KERNEL_RADIUS; ky++) {
            for (int kx = -KERNEL_RADIUS; kx <= KERNEL_RADIUS; kx++) {
                int nx = x + kx;
                int ny = y + ky;
                
                nx = max(0, min(width-1, nx));
                ny = max(0, min(height-1, ny));
                
                unsigned char pixel = input[(ny * width + nx) * channels + c];
                
                float kernelValue = kernel[(ky + KERNEL_RADIUS) * KERNEL_SIZE + 
                                          (kx + KERNEL_RADIUS)];
                
                sum += pixel * kernelValue;
            }
        }

        output[(y * width + x) * channels + c] = (unsigned char)sum;
    }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: %s <image_entree> <image_sortie>\n", argv[0]);
        return 1;
    }
    
    const char *inputPath = argv[1];
    const char *outputPath = argv[2];
    
    int width, height, channels;
    unsigned char *image = NULL;
    
    printf("Chargement de l'image: %s\n", inputPath);
    image = stbi_load(inputPath, &width, &height, &channels, 0);
    
    if (!image) {
        printf("Erreur lors du chargement de l'image\n");
        return 1;
    }
    
    printf("Image chargée: %dx%d avec %d canaux\n", width, height, channels);
    
    unsigned char *outputImage = (unsigned char*)malloc(width * height * channels);
    
    float *gaussianKernel = (float*)malloc(KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    createGaussianKernel(gaussianKernel, KERNEL_SIZE, SIGMA);
    
    unsigned char *d_input = NULL;
    unsigned char *d_output = NULL;
    float *d_kernel = NULL;
    
    cudaMalloc(&d_input, width * height * channels);
    cudaMalloc(&d_output, width * height * channels);
    cudaMalloc(&d_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    
    cudaMemcpy(d_input, image, width * height * channels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, gaussianKernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    
    printf("Lancement du kernel avec grid: %dx%d, block: %dx%d\n", 
           gridSize.x, gridSize.y, blockSize.x, blockSize.y);
    
    gaussianBlurKernel<<<gridSize, blockSize>>>(d_input, d_output, d_kernel, 
                                              width, height, channels);
    
    cudaDeviceSynchronize();
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    cudaMemcpy(outputImage, d_output, width * height * channels, cudaMemcpyDeviceToHost);
    
    printf("Sauvegarde de l'image floutée: %s\n", outputPath);
    if (strstr(outputPath, ".png")) {
        stbi_write_png(outputPath, width, height, channels, outputImage, width * channels);
    } else if (strstr(outputPath, ".jpg") || strstr(outputPath, ".jpeg")) {
        stbi_write_jpg(outputPath, width, height, channels, outputImage, 90); // Qualité 90%
    } else {
        printf("Format de sortie non supporté. Utilisez .png ou .jpg\n");
    }
    
    stbi_image_free(image);
    free(outputImage);
    free(gaussianKernel);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    
    printf("Traitement terminé avec succès !\n");
    
    return 0;
}