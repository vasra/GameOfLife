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

__global__ void nextGen(char* d_life, char* d_life_copy, int size) {
    // Shared memory grid
    extern __shared__ char sgrid[];

    int X = (blockIdx.x - 2) * blockDim.x + threadIdx.x;
    int Y = (blockIdx.y - 2) * blockDim.y + threadIdx.y;

    // The global ID of the thread in the grid
    int threadIdGlobal = (size + 2) * Y + X;

    // The local ID of the thread in the block
    int threadIdLocal = threadIdx.y * blockDim.x + threadIdx.x;

    int neighbours;

    if (X <= size + 1 && Y <= size + 1)
        sgrid[threadIdLocal] = d_life[threadIdGlobal];

    __syncthreads();

    // If the thread does not correspond to a halo element, then calculate its neighbours
    if(threadIdx.x > 0 && threadIdx.x < blockDim.x - 1 && threadIdx.y > 0 && threadIdx.y < blockDim.y - 1)
        neighbours = sgrid[threadIdLocal - blockDim.x - 1] + sgrid[threadIdLocal - blockDim.x]     + sgrid[threadIdLocal - blockDim.x + 1] +
                     sgrid[threadIdLocal - 1]               + /* you are here */                       sgrid[threadIdLocal + 1]               +
                     sgrid[threadIdLocal + blockDim.x - 1] + sgrid[threadIdLocal + blockDim.x]     + sgrid[threadIdLocal + blockDim.x + 1];

        if ((neighbours == 2 && sgrid[threadIdLocal] == 1) || (neighbours == 3))
            sgrid[threadIdLocal] == 1;
        else
            sgrid[threadIdLocal] == 0;
}