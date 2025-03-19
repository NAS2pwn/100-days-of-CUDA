#include <stdio.h>

__global__ void additionMatrices(int *a, int *b, int *c, int n, int m) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < m) {
        int idx = row * m + col;
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 4; // nombre de lignes
    int m = 4; // nombre de colonnes
    int size = n * m * sizeof(int);

    // Allocation de la mémoire sur le CPU (host)
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

    printf("Matrice A :\n");
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            printf("%2d ", h_a[i * m + j]);
        }
        printf("\n");
    }
    printf("\n");

    printf("Matrice B :\n");
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            printf("%2d ", h_b[i * m + j]);
        }
        printf("\n");
    }
    printf("\n");

    // Allocation de la mémoire sur le GPU (device)
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    /* EXPLICATION DE LA PARALLÉLISATION CUDA
     * 
     * Dans CUDA, le travail est organisé en une hiérarchie de threads :
     * 
     * 1. Threads :
     *    - C'est l'unité de base de l'exécution
     *    - Chaque thread exécute le même code (kernel) mais sur des données différentes
     *    - Identifié par threadIdx.x, threadIdx.y, threadIdx.z
     * 
     * 2. Blocs :
     *    - Groupe de threads qui peuvent coopérer entre eux
     *    - Peuvent partager de la mémoire
     *    - Identifié par blockIdx.x, blockIdx.y, blockIdx.z
     *    - Ici on utilise des blocs de 16x16 threads (dim3 threadsParBloc(16, 16))
     * 
     * 3. Grille :
     *    - Ensemble de blocs
     *    - Les blocs sont exécutés indépendamment
     * 
     * Pour notre matrice 4x4 avec des blocs de 16x16 threads :
     * - Même si on demande 16x16 threads par bloc, seuls 4x4 threads seront actifs
     * - La condition (row < n && col < m) dans le kernel évite le travail inutile
     * - La formule (m + threadsParBloc.x - 1) / threadsParBloc.x arrondit au supérieur
     *   pour s'assurer de couvrir toute la matrice
     * 
     * Calcul de l'index global :
     * row = blockIdx.y * blockDim.y + threadIdx.y
     * col = blockIdx.x * blockDim.x + threadIdx.x
     * 
     * Cette formule combine la position du bloc et la position du thread
     * pour obtenir la position unique de chaque thread dans la grille globale.
     */

    dim3 threadsParBloc(16, 16); // 256 threads par bloc
    dim3 numBlocs((m + threadsParBloc.x - 1) / threadsParBloc.x,
                  (n + threadsParBloc.y - 1) / threadsParBloc.y);

    additionMatrices<<<numBlocs, threadsParBloc>>>(d_a, d_b, d_c, n, m);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    printf("Matrice résultante C :\n");
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            printf("%2d ", h_c[i * m + j]);
        }
        printf("\n");
    }
    printf("\n");

    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
