#include <lcutil.h>
#include <timestamp.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <life.h>
#include <utility>

#define DEBUG

__global__ void nextGen(int rows, int columns, char* d_life, char* d_life_copy) {
    int neighbors = 0;

    for (int cellId = blockIdx.x * blockDim.x + threadIdx.x; cellId < rows * columns; cellId += blockDim.x * gridDim.x) {
            int x = cellId % columns;
            int yAbs = cellId - x;
            int xLeft = (x + columns - 1) % columns;
            int xRight = (x + 1) % columns;
            int yAbsUp = (yAbs + rows * columns - columns) % (rows * columns);
            int yAbsDown = (yAbs + columns) % (rows * columns);
            neighbors = d_life[xLeft + yAbsUp]   + d_life[x + yAbsUp]   + d_life[xRight + yAbsUp] +
                        d_life[xLeft + yAbs]               +                d_life[xRight + yAbs] +
                        d_life[xLeft + yAbsDown] + d_life[x + yAbsDown] + d_life[xRight + yAbsDown];

            if (neighbors == 3 || (neighbors == 2 && d_life_copy[x + yAbs] == 1))
                d_life_copy[x + yAbs] = 1;
            else
                d_life_copy[x + yAbs] = 0;
    }

}

//////////////////////////////////////////////////////////////////////////////////////
// Plays the Game Of Life. It checks the contents of d_life,
// calculates the results, and stores them in d_life_copy. The living organisms
// are represented by a 1, and the dead organisms by a 0.
//////////////////////////////////////////////////////////////////////////////////////
extern "C" float GameOfLife(int rows, int columns, char* h_life, char* h_life_copy, int nblocks, int nthreads, int generations) {
    // The grids that will be copied to the GPU
    char* d_life;
    char* d_life_copy;
    cudaError_t err;

    err = cudaMalloc((void**)&d_life, rows * columns * sizeof(char));
    if (err != cudaSuccess) {
        fprintf(stderr, "Could not allocate CUDA memory, with error code %d\n", err);
        return err;
    }

    err = cudaMalloc((void**)&d_life_copy, rows * columns * sizeof(char));
    if (err != cudaSuccess) {
        fprintf(stderr, "Could not allocate CUDA memory, with error code %d\n", err);
        return err;
    }

    err = cudaMemcpy(d_life, h_life, sizeof(char) * rows * columns, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Could not copy to GPU memory, with error code %d\n", err);
        return err;
    }

    err = cudaMemcpy(d_life_copy, h_life_copy, sizeof(char) * rows * columns, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Could not copy to GPU memory, with error code %d\n", err);
        return err;
    }

    timestamp t_start;
    t_start = getTimestamp();

    for (int gen = 0; gen < generations; gen++) {
        nextGen<<<nblocks, nthreads>>>(rows, columns, d_life, d_life_copy);

#ifdef DEBUG
        cudaMemcpy(h_life, d_life, sizeof(char) * rows * columns, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_life_copy, d_life_copy, sizeof(char) * rows * columns, cudaMemcpyDeviceToHost);
        printf("Generation %d\n", gen);
        printf("life\n");
        Print_grid(rows, columns, h_life);
        printf("life_copy\n");
        Print_grid(rows, columns, h_life_copy);
#endif
        /////////////////////////////////////////////////////////////////////////////////////////////////
        // Swap the addresses of the two tables. That way, we avoid copying the contents
        // of d_life to d_life_copy. Each round, the addresses are exchanged, saving time from running
        // a loop to copy the contents.
        /////////////////////////////////////////////////////////////////////////////////////////////////
        std::swap(d_life, d_life_copy);
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error synchronizing devices: %s\n", err);
        return err;
    }
    
    float msecs = getElapsedtime(t_start);

    err = cudaFree(d_life);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error freeing GPU memory: %s\n", err);
        return err;
    }

    err = cudaFree(d_life_copy);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error freeing GPU memory: %s\n", err);
        return err;
    }

    return msecs;
}