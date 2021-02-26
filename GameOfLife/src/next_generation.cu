#include <gol.cuh>

//#define DEBUG

__global__ void copyRowHalos(char* d_life, int size) {
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadID <= size + 1) {
        d_life[threadID] = d_life[size * (size + 2)  + threadID];   // copy bottom row to upper halo row
        d_life[size + 1 + threadID] = d_life[size + 3 + threadID];  // copy upper row to bottom halo row
    }
}

__global__ void copyHaloColumns(char* d_life, int size) {
    int threadID = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (threadID <= size + 1) {
        d_life[threadID * (size + 3)] = d_life[size * 2 * threadID];       // copy rightmost column to left halo column
        d_life[(size + 1) * 2 * threadID] = d_life[(size + 2) * threadID]; // copy leftmost column to right halo column
    }
}

__global__ void nextGen(char* d_life, char* d_life_copy, const int size) {
    // The rows and columns of the local grid
    __shared__ char* lgrid = (char*)malloc(size * size * sizeof(char));
    __shared__ char* lgridCopy = (char*)malloc(size * size * sizeof(char));
}