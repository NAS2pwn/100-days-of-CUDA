#include <stdio.h>

__global__ void additionMatrices(int *a, int *b, int *c, int n, int m) {
    int idx = threadIdx.x;
    c[idx] = a[idx] + b[idx];
}

void printMatrice(const char *name, int *matrice, int n, int m) {
    printf("%s :\n", name);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            printf("%2d ", matrice[i * m + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    int n = 4;
    int m = 4;
    int size = n * m * sizeof(int);

    int h_a[16] = {
        1, 0, 2, -1,
        3, 4, -2, 5,
        1, 1, 1, 1,
        0, 2, 3, -4
    };

    int h_b[16] = {
        2, 1, 0, 1,
        -1, 2, 3, 0,
        1, -1, 2, 2,
        3, 0, -2, 1
    };

    int *h_c = (int*)malloc(size);

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    dim3 threadsParBloc(16);
    dim3 numBlocs(1);

    additionMatrices<<<numBlocs, threadsParBloc>>>(d_a, d_b, d_c, n, m);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    printMatrice("Matrice A", h_a, n, m);
    printMatrice("Matrice B", h_b, n, m);
    printMatrice("Matrice C", h_c, n, m);

    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}