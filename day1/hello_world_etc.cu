#include <stdio.h>

// __global__ signifie qu'il s'agit d'un kernel CUDA
__global__ void afficherThreadID() {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Thread ID : %d\n", id);
}

int main() {
    printf("Hello World !\n");
    int nbBlocs = 2;
    int threadsParBloc = 5;

    afficherThreadID<<<nbBlocs, threadsParBloc>>>();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Erreur de lancement: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Erreur d'exécution: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}
