#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <life.h>
#include <random>

//#define DEBUG

///////////////////////////////////////////////////////////////////////
// size        - The size of one side of the square grid
// generations - The number of generations for which the game will run
// nthreads    - The number of threads per block
// dimGr     - The number of blocks
///////////////////////////////////////////////////////////////////////
#ifndef DEBUG
constexpr int size = 840;
constexpr int generations = 5;
constexpr int nthreads = 64;
constexpr int dimGr = size * size / nthreads;
const     int blockSide = static_cast<int>(size / sqrt(dimGr));
dim3 dimBl(blockSide, blockSide);
#else
constexpr int size = 8;
constexpr int generations = 2;
constexpr int nthreads = 4;
constexpr int dimGr = size * size / nthreads;
const     int blockSide = static_cast<int>(size / sqrt(dimGr));
dim3 dimBl(blockSide, blockSide);
#endif

int main() {
    /////////////////////////////////////////////////////////////////////////////
    // rows    - The number of rows of the 2D matrix
    // columns - The number of columns of the 2D matrix
    // Having two variables be the same as the size variable may seem redundant,
    // but it increases code readability
    /////////////////////////////////////////////////////////////////////////////
    int rows, columns;
    rows = columns = size;

    // Pointers to our 2D grid, and its necessary copy
    char *h_life = (char*)malloc( rows * columns * sizeof(char) );
    char *h_life_copy = (char*)malloc( rows * columns * sizeof(char) );

    // Produce the first generation randomly
    Initial_state(rows, columns, h_life, h_life_copy);
    
    float msecs = GameOfLife(rows, columns, h_life, h_life_copy, dimGr, dimBl, generations);

    printf("Elapsed time is %.2f msecs\n", msecs);

    // Clean up and exit
    free(h_life);
    free(h_life_copy);
 
    return 0;
}

/////////////////////////////////////////////////////////////////
// Randomly produces the first generation. The living organisms
// are represented by a 1, and the dead organisms by a 0.
/////////////////////////////////////////////////////////////////
void Initial_state(int rows, int columns, char * h_life, char * h_life_copy) {
    float randomProbability = 0.0f;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> probability(0.0f, 1.0f);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            randomProbability = static_cast<float>(probability(gen));
            if (randomProbability >= 0.5f)
                *(h_life + i * columns + j) = *(h_life_copy + i * columns + j) = 1;
            else
                *(h_life + i * columns + j) = *(h_life_copy + i * columns + j) = 0;
        }
    }
}

/////////////////////////////////////////////////////////////////
// Prints the entire grid to the terminal. Used for debugging
/////////////////////////////////////////////////////////////////
void Print_grid(int rows, int columns, char * h_life) {
    for (int i = 0; i< rows; i++) {
        for (int j = 0; j < columns; j++) {
            printf("%d ", *(h_life + i * columns + j));
            if ( j == columns - 1)
                printf("\n");
        }
    }
    printf("\n");
}