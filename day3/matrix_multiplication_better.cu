#include <stdio.h>

#define N 5000

__global__ void matrixMulBetter(float* A, float* B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

void printMatrixPartially(float* matrix, const char* name, int n_visible_rows, int n_visible_cols) {
    printf("\nMatrix %s (showing %d x %d elements):\n", name, n_visible_rows, n_visible_cols);
    
    for (int i = 0; i < n_visible_rows-1; i++) {
        for (int j = 0; j < n_visible_cols-1; j++) {
            printf("%.1f\t", matrix[i * N + j]);
        }
        printf("...\t");
        printf("%.1f\n", matrix[i * N + (N-1)]);
    }
    printf("...\n");

    for (int j = 0; j < n_visible_cols-1; j++) {
        printf("%.1f\t", matrix[(N-1) * N + j]);
    }
    printf("...\t");
    printf("%.1f\n", matrix[(N-1) * N + (N-1)]);
}

int main() {
    float *h_A = (float*)malloc(N * N * sizeof(float));
    float *h_B = (float*)malloc(N * N * sizeof(float));
    float *h_C = (float*)malloc(N * N * sizeof(float));

    for (int i = 0; i < N * N; i++) {
        h_A[i] = i + 1;
        if (i/N == i%N) {
            h_B[i] = 2;
        } else {
            h_B[i] = 0;
        }
    }

    printMatrixPartially(h_A, "A", 5, 5);
    printMatrixPartially(h_B, "B", 5, 5);

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
    (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixMulBetter<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    printMatrixPartially(h_C, "C", 5, 5);

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}