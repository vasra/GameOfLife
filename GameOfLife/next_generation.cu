#include <lcutil.h>
#include <timestamp.h>
#include <stdio.h>
#include <stdlib.h>
#include <utility>

#define DEBUG_COORDINATES
#define DEBUG_GRID

__global__ void nextGen(int rows, int columns, char* lifeCUDA, char* lifeCUDA_copy) {
    int neighbors = 0;

    for (int cellId = blockIdx.x * blockDim.x + threadIdx.x; cellId < rows * columns; cellId += blockDim.x * gridDim.x) {
            int x = cellId % columns;
            int yAbs = cellId - x;
            int xLeft = (x + columns - 1) % columns;
            int xRight = (x + 1) % columns;
            int yAbsUp = (yAbs + rows * columns - columns) % (rows * columns);
            int yAbsDown = (yAbs + columns) % (rows * columns);
            neighbors = lifeCUDA[xLeft + yAbsUp]   + lifeCUDA[x + yAbsUp]   + lifeCUDA[xRight + yAbsUp] +
                        lifeCUDA[xLeft + yAbs]               +                lifeCUDA[xRight + yAbs] +
                        lifeCUDA[xLeft + yAbsDown] + lifeCUDA[x + yAbsDown] + lifeCUDA[xRight + yAbsDown];

            if (neighbors == 3 || (neighbors == 2 && lifeCUDA_copy[x + yAbs] == 1))
                lifeCUDA_copy[x + yAbs] = 1;
            else
                lifeCUDA_copy[x + yAbs] = 0;
    }
}

//////////////////////////////////////////////////////////////////////////////////////
// Plays the Game Of Life. It checks the contents of lifeCUDA,
// calculates the results, and stores them in lifeCUDA_copy. The living organisms
// are represented by a 1, and the dead organisms by a 0.
//////////////////////////////////////////////////////////////////////////////////////
extern "C" float GameOfLife(int rows, int columns, char* life, char* life_copy, int nblocks, int nthreads, int generations) {
    cudaError_t err;
    char* lifeCUDA;
    char* lifeCUDA_copy;

    err = cudaMalloc((void**)&lifeCUDA, rows * columns * sizeof(char));
    if (err != cudaSuccess) {
        fprintf(stderr, "Could not allocate CUDA memory, with error code %d\n", err);
        return err;
    }

    err = cudaMalloc((void**)&lifeCUDA_copy, rows * columns * sizeof(char));
    if (err != cudaSuccess) {
        fprintf(stderr, "Could not allocate CUDA memory, with error code %d\n", err);
        return err;
    }

    err = cudaMemcpy(lifeCUDA, life, sizeof(char) * rows * columns, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Could not copy to GPU memory, with error code %d\n", err);
        return err;
    }

    err = cudaMemcpy(lifeCUDA_copy, life_copy, sizeof(char) * rows * columns, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Could not copy to GPU memory, with error code %d\n", err);
        return err;
    }

    timestamp t_start;
    t_start = getTimestamp();

    for (int i = 0; i < generations; i++) {
        nextGen<<<nblocks, nthreads>>>(rows, columns, lifeCUDA, lifeCUDA_copy);
#ifdef DEBUG_GRID
        // Print the grid of every block, before the exchange of the halo elements and before the beginning of the main loop

#endif

#ifdef DEBUG_GRID
        // Print the grid of every block          
#endif
        /////////////////////////////////////////////////////////////////////////////////////////////////
        // Swap the addresses of the two tables. That way, we avoid copying the contents
        // of lifeCUDA to lifeCUDA_copy. Each round, the addresses are exchanged, saving time from running
        // a loop to copy the contents.
        /////////////////////////////////////////////////////////////////////////////////////////////////
        std::swap(lifeCUDA, lifeCUDA_copy);
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error synchronizing devices: %s\n", err);
        return err;
    }
    
    float msecs = getElapsedtime(t_start);

    err = cudaFree(lifeCUDA);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error freeing GPU memory: %s\n", err);
        return err;
    }

    err = cudaFree(lifeCUDA_copy);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error freeing GPU memory: %s\n", err);
        return err;
    }

    return msecs;
}