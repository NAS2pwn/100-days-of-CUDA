#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdio.h>

// Ajout de extern "C" pour l'interfaçage avec Python
extern "C" {
    void compute_dft(cuDoubleComplex* h_input, cuDoubleComplex* h_output, int N);
}

__global__ void dft_kernel(cuDoubleComplex* x, cuDoubleComplex* X, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < N) {
        cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);

        for (int n = 0; n < N; n++) {
            double angle = -2.0 * M_PI * k * n / N;
            cuDoubleComplex w = make_cuDoubleComplex(cos(angle), sin(angle));
            sum = cuCadd(sum, cuCmul(x[n], w));
        }

        X[k] = sum;
    }
}

void compute_dft(cuDoubleComplex* h_input, cuDoubleComplex* h_output, int N) {
    cuDoubleComplex *d_input, *d_output;

    cudaMalloc(&d_input, N * sizeof(cuDoubleComplex));
    cudaMalloc(&d_output, N * sizeof(cuDoubleComplex));

    cudaMemcpy(d_input, h_input, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    dft_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);

    cudaMemcpy(h_output, d_output, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}