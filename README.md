# 100-days-of-CUDA
Just building CUDA kernel so I learn !

## Day 1

Objective :

- [x] Compile and run my first CUDA kernel
- [x] Read chapter 1 of the CUDA book so I understand why CUDA is used today

Implemented :

- `hello_world_etc.cu` : Hello world kernel, every thread greets the user
- `add_matrices_bad.cu` : Simple kernel to add two matrices, Ai generated SLOP code but icebreaker

Learnt :

- For my GPU (turing architecture), I need to add the `-arch=sm_75` flag to the compilation command.

- Understood the GPU programming background

- Do not still understand the GPU architecture

Time spent : 20 minutes

## Day 2

Objective :

- [x] Read what I missed in the chapter 1 of the CUDA book
- [x] Understood threads and blocks

Implemented :

- `introspection.cu` : Just displaying useful information about the GPU
- `add_matrices_better.cu` : Better kernel to add two matrices, implemented manually, optimized
- `matrix_multiplication_naive.cu` : Pretty basic matrix multiplication implementation with shared memory

Learnt :

- Better understanding of threads, blocks, warp, memory...
- Better understanding of challenges of GPU programming
- Hands-on experience with shared memory and thread synchronization

Time spent : 50 minutes

## Day 3

Objective :

- [x] Implementing a larger matrix multiplication
- [x] Implementing a convolution operation
- [x] Using nvidia nsight for the first time

Implemented :

- `matrix_multiplication_better.cu` : Better matrix multiplication, on a real word sized matrix
- `gaussian_blur_vibe_coded.cu` : Gaussian blur kernel coded with cursor

Learnt :

- Learnt how to use 100% of my GPU
- Used `sudo ncu -o profile ./binary` to profile the kernel, it's pretty cool
- Implemented a gaussian blur kernel, but I used cursor to write the code so I don't really understand what happens, but you know I have some insights

Time spent : 1 hour

## Day 4

Objective :

- [] Fully understand the gaussian blur kernel
- [] Develop another operation on image manually
- [] Read a little bit of the CUDA book

Implemented :


Time spent : 