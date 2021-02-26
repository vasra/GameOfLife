#include <gol.cuh>

//#define DEBUG

__global__ void copyHaloRows(char* d_life, const int size) {
    int threadID = blockIdx.x * blockDim.x + threadIdx.x + 1;

    // threadID must be in the range [1, size]
    if (threadID <= size) {
        d_life[threadID] = d_life[size * (size + 2) + threadID];                   // copy bottom row to upper halo row
        d_life[(size + 2) * (size + 1) + threadID] = d_life[size + 2 + threadID];  // copy upper row to bottom halo row
    }
}

__global__ void copyHaloColumns(char* d_life, const int size) {
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    
    // threadID must be in the range [0, size + 1]
    if (threadID <= size + 1) {
        d_life[(size + 2) * threadID] = d_life[(size + 2) * threadID + size];           // copy rightmost column to left halo column
        d_life[(size + 1) * threadID + (size + 1)] = d_life[(size + 2) * threadID + 1]; // copy leftmost column to right halo column
    }
}

__global__ void nextGen(char* d_life, char* d_life_copy, dim3 gridSize) {
    // Shared memory grid
    extern __shared__ char sgrid[];

    // The global x and y coordinates of the thread
    int x = blockId.x * blockDim.x + threadId.x;
    int y = blockId.y * blockDim.y + threadId.y;

    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    int neighbours;

    if (threadID <= (size + 2) * (size + 2))
        sgrid[threadID] = d_life[threadID];

    syncthreads();

    if((threadID > size + 2) && (threadID < )
        neighbours = sgrid[threadID - size - 3] + sgrid[threadID - size - 2] + sgrid[threadID - size - 1] +
                     sgrid[threadID - 1]        + /* you are here */           sgrid[threadID + 1]        +
                     sgrid[threadID + size - 1] + sgrid[threadID + size]     + sgrid[threadID + size + 1];


    free(sgrid);
}