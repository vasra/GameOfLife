#include <lcutil.h>
#include <timestamp.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <life.h>
#include <utility>

//#define DEBUG

__global__ void nextGen(const int rows, const int columns, char* d_life, char* d_life_copy) {
    int neighbors = 0;
    int first_in_row, down, up, left, right, upright, upleft, downright, downleft;
    
    for (int cell = blockIdx.x * blockDim.x + threadIdx.x; cell < rows * columns; cell += blockDim.x * gridDim.x) {
            first_in_row = cell -  cell % columns;
            down         = (cell + columns) % (rows * columns);
            up           = (cell + rows * columns - columns) % (rows * columns);
            left         = (cell + rows * columns - 1) % columns + first_in_row;
            right        = (cell + rows * columns + 1) % columns + first_in_row;
            upleft       = (left + rows * columns - columns) % (rows * columns);
            downleft     = (left + rows * columns + columns) % (rows * columns);
            upright      = (right + rows * columns - columns) % (rows * columns);
            downright    = (right + rows * columns + columns) % (rows * columns);
#ifdef DEBUG       
            if (cell == 0 || cell ==4 || cell == 20 || cell == 24 || cell == 12)
                printf("I am cell %d and my neighbors are up %d down %d right %d left %d upright %d upleft %d downright %d downleft %d\n", cell, up, down, right, left, upright, upleft, downright, downleft);
#endif
            neighbors = *(d_life + upleft)   + *(d_life + up)   + *(d_life + upright) +
                        *(d_life + left)                +         *(d_life + right)   +
                        *(d_life + downleft) + *(d_life + down) + *(d_life + downright);
        
            if (neighbors == 3 || (neighbors == 2 && *(d_life_copy + cell) == 1))
                *(d_life_copy + cell) = 1;
            else
                *(d_life_copy + cell) = 0;
    }
}

//////////////////////////////////////////////////////////////////////////////////////
// Plays the Game Of Life. It checks the contents of d_life,
// calculates the results, and stores them in d_life_copy. The living organisms
// are represented by a 1, and the dead organisms by a 0.
//////////////////////////////////////////////////////////////////////////////////////
extern "C" float GameOfLife(const int rows, const int columns, char* h_life, char* h_life_copy, dim3 dimGr, dim3 dimBl, int generations) {
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
        nextGen<<<dimGr, dimBl>>>(rows, columns, d_life, d_life_copy);

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