#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <random>

void Initial_state(int rows, int columns, char *first_generation, char *first_generation_copy);
void Print_grid(int rows, int columns, char *life);
extern "C" float GameOfLife(int rows, int columns, char* life, char* life_copy, int nblocks, int nthreads, int generations);

// The size of one side of the square grid
constexpr int size = 2048;
constexpr int generations = 5;
constexpr int nblocks = 5;
constexpr int nthreads = 256;

int main() {
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    // rows    - The number of rows of the local 2D matrix of the block
    // columns - The number of columns of the local 2D matrix of the block
    /////////////////////////////////////////////////////////////////////////////////////////////////////////

    int rows, columns;

    // We add 2 to each dimension in order to include the halo rows and columns
    rows = columns = size + 2;

    // Pointers to our 2D grid, and its necessary copy
    char *life = (char*)malloc( rows * columns * sizeof(char) );
    char *life_copy = (char*)malloc( rows * columns * sizeof(char) );

    // Produce the first generation
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
void Initial_state(int rows, int columns, char *first_generation, char *first_generation_copy) {
    // Generate the first generation according to the random seed
    float randomProbability = 0.0f;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> probability(0.0f, 1.0f);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            // Initialize all halo values to 0. The rest will be assigned values randomly
            if ( i == 0 || j == 0 || i == rows - 1 || j == columns - 1) {
                *(first_generation + i * columns + j) = *(first_generation_copy + i * columns + j) = 0;
                continue;
            }
            randomProbability = static_cast<float>(probability(gen));
            if (randomProbability >= 0.5f)
                *(first_generation + i * columns + j) = *(first_generation_copy + i * columns + j) = 1;
            else
                *(first_generation + i * columns + j) = *(first_generation_copy + i * columns + j) = 0;
        }
    }
}

/////////////////////////////////////////////////////////////////
// Prints the entire grid to the terminal. Used for debugging
/////////////////////////////////////////////////////////////////
void Print_grid(int rows, int columns, char *life) {
    for (int i = 0; i< rows; i++) {
        for (int j = 0; j < columns; j++) {
            printf("%d ", *(life + i * columns + j));
            if ( j == columns - 1)
                printf("\n");
        }
    }
    printf("\n");
}