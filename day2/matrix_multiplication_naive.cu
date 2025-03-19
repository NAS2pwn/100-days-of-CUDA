#include <stdio.h>

#define N 4 // 4x4 matrix

__global__ void matrixMulNaive(float* A, float* B, float* C) {
    // Only one block for now
    int row = threadIdx.y;
    int col = threadIdx.x;

    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

void printMatrix(float* matrix, const char* name) {
    printf("\nMatrice %s:\n", name);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.1f\t", matrix[i * N + j]);
        }
        printf("\n");
    }
}

int main() {
    float *h_A = (float*)malloc(N * N * sizeof(float));
    float *h_B = (float*)malloc(N * N * sizeof(float));
    float *h_C = (float*)malloc(N * N * sizeof(float));
    
    for (int i = 0; i < N * N; i++) {
        h_A[i] = i + 1;
        h_B[i] = i * 2;
    }

    printMatrix(h_A, "A");
    printMatrix(h_B, "B");

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(N, N);
    matrixMulNaive<<<1, threadsPerBlock>>>(d_A, d_B, d_C);

    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    printMatrix(h_C, "C (Result)");

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}