#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <random>

#define DEBUG_COORDINATES
#define DEBUG_GRID

void Initial_state(int rows, int columns, char *first_generation, char *first_generation_copy);
void Print_grid(int rows, int columns, char *life);
__global__ void Next_generation(int rows, int columns, char *lifeCUDA, char *lifeCUDA_copy);
void inline Swap(char **a, char **b);

// The size of one side of the square grid
constexpr int SIZE = 8;
constexpr int NDIMS = 2;

int main()
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    // VARIABLES FOR THE CARTESIAN TOPOLOGY
    // rows    - The number of rows of the local 2D matrix
    // columns - The number of columns of the local 2D matrix
    // seed    - The seed used to randomly create the first generation
    /////////////////////////////////////////////////////////////////////////////////////////////////////////

    int rows, columns;
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // VARIABLES
    // t1, t2            - Used for timing
    // root              - Used to check if the number of processes is a perfect square
    // randomProbability - Used to randomly produce the first generation
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    double t1, t2;

    // We add 2 to each dimension in order to include the halo rows and columns
    rows = columns = SIZE + 2;

    // Pointers to our 2D grid, and its necessary copy
    char *life = (char*)malloc( rows * columns * sizeof(char) );
    char *life_copy = (char*)malloc( rows * columns * sizeof(char) );
    char* lifeCUDA;
    char* lifeCUDA_copy;
    cudaError_t err;

    // Produce the first generation
    Initial_state(rows, columns, life, life_copy);

    err = cudaMalloc((void**)&lifeCUDA, rows * columns * sizeof(char));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s\n", err);
        return err;
    }

    err = cudaMalloc((void**)&lifeCUDA_copy, rows * columns * sizeof(char));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s\n", err);
        return err;
    }

    err = cudaMemcpy(lifeCUDA, life, sizeof(char) * rows * columns, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s\n", err);
        return err;
    }

    err = cudaMemcpy(lifeCUDA_copy, life_copy, sizeof(char) * rows * columns, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s\n", err);
        return err;
    }

#ifdef DEBUG_GRID
    // Print the grid of every block, before the exchange of the halo elements and before the beginning of the main loop
    
#endif

    // Modify the number of generations as desired
    for (int generation = 0; generation < 5; generation++)
    {
        Next_generation(rows, columns, life, life_copy);

#ifdef DEBUG_GRID
        // Print the grid of every block (very few threads must be used)          
#endif
        /////////////////////////////////////////////////////////////////////////////////////////////////
        // Swap the addresses of the two tables. That way, we avoid copying the contents
        // of life to life_copy. Each round, the addresses are exchanged, saving time from running
        // a loop to copy the contents.
        /////////////////////////////////////////////////////////////////////////////////////////////////
        Swap(&life, &life_copy);
    }
    // Clean up and exit
    free(life);
    free(life_copy);
    cudaFree(lifeCUDA);
    cudaFree(lifeCUDA_copy);
    return 0;
}

/////////////////////////////////////////////////////////////////
// Randomly produces the first generation. The living organisms
// are represented by a 1, and the dead organisms by a 0.
/////////////////////////////////////////////////////////////////
void Initial_state(int rows, int columns, char *first_generation, char *first_generation_copy)
{
    // Generate the first generation according to the random seed
    float randomProbability = 0.0f;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> probability(0.0f, 1.0f);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            // Initialize all halo values to 0. The rest will be assigned values randomly
            if ( i == 0 || j == 0 || i == rows - 1 || j == columns - 1)
            {
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
void Print_grid(int rows, int columns, char *life)
{
    for (int i = 0; i< rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            printf("%d ", *(life + i * columns + j));
            if ( j == columns - 1)
                printf("\n");
        }
    }
    printf("\n");
}

//////////////////////////////////////////////////////////////////////////////////////
// Produces the next generation. It checks the contents of lifeCUDA,
// calculates the results, and stores them in lifeCUDA_copy. The living organisms
// are represented by a 1, and the dead organisms by a 0.
//////////////////////////////////////////////////////////////////////////////////////
__global__ void Next_generation(int rows, int columns, char *lifeCUDA, char *lifeCUDA_copy)
{
    int neighbors;
    
    for (int i = 2; i < rows - 2; i++)
    {
        for (int j = 2; j < columns - 2; j++)
        {
            neighbors = *(lifeCUDA + (i - 1) * columns + (j - 1)) + *(lifeCUDA + (i - 1) * columns + j) + *(lifeCUDA + (i - 1) * columns + (j + 1)) +
                        *(lifeCUDA + i * columns + (j - 1))                          +                *(lifeCUDA + i * columns + (j + 1))       +
                        *(lifeCUDA + (i + 1) * columns + (j - 1)) + *(lifeCUDA + (i + 1) * columns + j) + *(lifeCUDA + (i + 1) * columns + (j + 1));

            if (neighbors == 3 || (neighbors == 2 && *(lifeCUDA_copy + i * columns + j) == 1))
                *(lifeCUDA_copy + i * columns + j) = 1;
            else
                *(lifeCUDA_copy + i * columns + j) = 0;
        }
    }
}

void inline Swap(char **a, char **b)
{
    char *temp = *a;
    *a = *b;
    *b = temp;
}
