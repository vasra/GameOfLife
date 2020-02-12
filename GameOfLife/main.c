#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"

#define SIZE 16
#define NDIMS 2
#define PROCESSES 4

void initial_state(char first_generation[SIZE][SIZE], char first_generation_copy[SIZE][SIZE]);
void print_grid(char life[SIZE][SIZE]);
void next_generation(char current_generation[SIZE][SIZE], char current_generation_copy[SIZE][SIZE]);
void swap(char (**a)[SIZE], char (**b)[SIZE]);

int main()
{
    char life[SIZE][SIZE];
    char life_copy[SIZE][SIZE];
    char (*current_generation)[SIZE] = life;
    char (*current_generation_copy)[SIZE] = life_copy;

    int dim_size[NDIMS], periods[NDIMS], reorder, rank, nrow, ncol;
    MPI_Comm cartesian2D;

    /*
     *
     * MISTAKE! FIX!
     */
    if( PROCESSES * PROCESSES == SIZE )
        dims[0] = dims[1] = PROCESSES;
    else
    {
        dims[0] = SIZE / PROCESSES;
        dims[1] = dims[0] / 2;
    }

    /* Our Cartesian topology will be a torus, so both fields of "periods" will have value of 1 */
    periods[0] = 1;
    periods[1] = 1;

    /* We allow MPI to efficiently reorder the processes among the different processors. */
    reorder = 1;

    /* Seed the random number generator, and generate the first generation according to that. */
    initial_state(current_generation, current_generation_copy);

    printf("------------------------------------\n");
    printf("    Welcome to the Game Of Life!\n");
    printf("------------------------------------\n\n");

    print_grid(current_generation);

    MPI_Init(NULL, NULL);
    MPI_Cart_create(MPI_COMM_WORLD, NDIMS, dims, periods, reorder, &cartesian2D);
    MPI_Comm_size(cartesian2D, &processes);
    /*
     * The Game Of Life will run for 5 generations. Modify the number
     * of generations as desired.
     */
    for(int i = 0; i < 4; i++)
    {
        next_generation(current_generation, current_generation_copy);
        print_grid(current_generation);

        /*
         * Swap the addresses of the two tables. That way, we avoid copying the contents
         * of current_generation to current_generation_copy in the next_generation function.
         * Each round, the addresses are exchanged, saving time from running a loop to copy the contents.
         */
        swap(&current_generation, &current_generation_copy);
    }
    return 0;
}

/* Randomly generates the first generation. The living organisms
 * are represented by an 1, and the dead organisms by a 0.
 */
void initial_state(char first_generation[SIZE][SIZE], char first_generation_copy[SIZE][SIZE])
{
    float probability;
    srand(time(NULL));

    for(int i = 0; i < SIZE; i++)
    {
        for(int j = 0; j < SIZE; j++)
        {
            probability = (float)rand() / (float)((unsigned)RAND_MAX + 1);
            if(probability >= 0.5f)
                first_generation[i][j] = first_generation_copy[i][j] = 1;
            else
                first_generation[i][j] = first_generation_copy[i][j] = 0;
        }
    }
}

void print_grid(char life[SIZE][SIZE])
{
    for (int i = 0; i< SIZE; i++)
    {
        for(int j = 0; j < SIZE; j++)
        {
            printf("%d ", life[i][j]);
            if( j == SIZE - 1)
                printf("\n");
        }
    }
    printf("\n");
}

/*
 * Produces the next generation. It checks the contents of current_generation_copy,
 * calculates the results, and stores them in current_generation. The living organisms
 * are represented by a 1, and the dead organisms by a 0.
 */
void inline next_generation(char current_generation[SIZE][SIZE], char current_generation_copy[SIZE][SIZE])
{
    int neighbors;

    for(int i = 0; i < SIZE; i++)
    {
        for(int j = 0; j < SIZE; j++)
        {
            neighbors = 0;
            for(int k = i - 1; k <= i + 1; k++)
            {
                for(int l = j - 1; l <= j + 1; l++)
                {
                    if(k > -1 && k < SIZE && l > -1 && l < SIZE)
                    {
                        if(current_generation_copy[k][l] == 1)
                            neighbors++;
                    }
                }
            }
            if(current_generation_copy[i][j] == 1)
                neighbors--;

            if(neighbors == 3 || (neighbors == 2 && current_generation_copy[i][j] == 1))
                current_generation[i][j] = 1;
            else
                current_generation[i][j] = 0;
        }
    }

    /*
     * This nested loop is not needed anymore. Instead of copying the contents of current_generation
     * into current_generation_copy every round, we simply swap their addresses

    for(int i = 0; i < SIZE; i++)
    {
        for(int j = 0; j < SIZE; j++)
        {
            current_generation_copy[i][j] = current_generation[i][j];
        }
    }
     */
}

void inline swap(char (**a)[SIZE], char (**b)[SIZE])
{
    char (*temp)[SIZE] = *a;
    *a = *b;
    *b = temp;
}
