#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <life.h>
#include <random>

#define DEBUG

// The size of one side of the square grid
#ifndef DEBUG
constexpr int size = 2048;
constexpr int generations = 5;
constexpr int nblocks = 5;
constexpr int nthreads = 256;
#else
constexpr int size = 8;
constexpr int generations = 2;
constexpr int nblocks = 1;
constexpr int nthreads = 2;
#endif

int main() {
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    // rows    - The number of rows of the local 2D matrix of the block
    // columns - The number of columns of the local 2D matrix of the block
    /////////////////////////////////////////////////////////////////////////////////////////////////////////

    int rows, columns;
    rows = columns = size;

    // Pointers to our 2D grid, and its necessary copy
    char *h_life = (char*)malloc( rows * columns * sizeof(char) );
    char *h_life_copy = (char*)malloc( rows * columns * sizeof(char) );

    // Produce the first generation randomly
    Initial_state(rows, columns, life, life_copy);
    
    float msecs = GameOfLife(rows, columns, life, life_copy, nblocks, nthreads, generations);

    printf("Elapsed time is %.2f msecs\n", msecs);

    // Clean up and exit
    free(life);
    free(life_copy);
 
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