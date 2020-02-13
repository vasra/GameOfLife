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
    /**< Our 2D grid in memory, and its necessary copy. */
    char life[SIZE][SIZE];
    char life_copy[SIZE][SIZE];

    /**< Pointers to the above grids. */
    char (*current_generation)[SIZE] = life;
    char (*current_generation_copy)[SIZE] = life_copy;

    /*******************************************************************************
     * ARRAYS FOR THE CARTESIAN TOPOLOGY
     * dim_size - Array with two elements
     *     dim_size[0] - How many processes will be in each row
     *     dim_size[1] - How many processes will be in each column
     *
     * periods  - Array with two elements, for the periodicity of the two dimensions
     *
     * coords   - Array with two elements, holding the coordinates of each process
     *******************************************************************************/

    int dim_size[NDIMS], periods[NDIMS], coords[NDIMS];

    /*******************************************************************************************************
     * VARIABLES FOR THE CARTESIAN TOPOLOGY
     * reorder         - Indicates if MPI can rearrange the processes more efficiently among the processors
     * rank            - Process rank
     * my_row_index    - Where each process starts in the rows of the 2D matrix
     * my_column_index - Where each process starts in the columns of the 2D matrix
     * cartesian2D     - Our new custom Communicator
     *******************************************************************************************************/

    int reorder, rank, my_row_index, my_column_index;
    MPI_Comm cartesian2D;

    /**< Our Cartesian topology will be a torus, so both fields of "periods" array will have a value of 1. */
    periods[0] = 1;
    periods[1] = 1;

    /**< We will allow MPI to efficiently reorder the processes among the different processors. */
    reorder = 1;

     /**< Seed the random number generator, and generate the first generation according to that. */
    initial_state(current_generation, current_generation_copy);

    printf("------------------------------------\n");
    printf("    Welcome to the Game Of Life!\n");
    printf("------------------------------------\n\n");

    print_grid(current_generation);

    /**< Initialize MPI */
    MPI_Init(NULL, NULL);

    /**< First, find the rank of each process in the default communicator. */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /**< Let MPI decide which is the best arrangement according to the number of processes and dimensions. */
    MPI_Dims_create(PROCESSES, NDIMS, dim_size);

    /**< Create a 2D Cartesian topology. */
    MPI_Cart_create(MPI_COMM_WORLD, NDIMS, dim_size, periods, reorder, &cartesian2D);

    /**< Synchronize all the processes before we start. */
    MPI_Barrier(cartesian2D);

    /**< We must find the rank of each process. This must run only once, process 0 can handle it. */
    if(rank == 0)
    {
        printf("There are %d dimensions and %d processes\n", NDIMS, PROCESSES);
        for(i = 0; i < dim_size[0]; i++)
        {
            for(int j = 0; j < dim_size[1]; j++)
            {
                printf("My rank is %d\n", rank);
                coords[0] = i;
                coords[1] = j;
                MPI_Cart_rank(cartesian2D, coords, &rank);
                printf("The rank for coords[%d][%d] is %d\n", i, j, rank);
            }
        }
    }

    my_row_index    = rank * (SIZE / dim_size[0]);
    my_column_index = rank * (SIZE / dim_size[1]);
    printf("Process %d: My row index is %d and my column index is %d\n", rank, my_row_index, my_column_index);

    /***********************************************
     * The Game Of Life will run for 5 generations.
     * Modify the number of generations as desired.
     ***********************************************/
//    for(int i = 0; i < 4; i++)
//    {
//        next_generation(current_generation, current_generation_copy);
//        print_grid(current_generation);

        /************************************************************************************************
         * Swap the addresses of the two tables. That way, we avoid copying the contents
         * of current_generation to current_generation_copy in the next_generation function.
         * Each round, the addresses are exchanged, saving time from running a loop to copy the contents.
         ************************************************************************************************/
//        swap(&current_generation, &current_generation_copy);
//    }

    /**< Clean up and exit. */
    MPI_Finalize();
    return 0;
}

/****************************************************************
 * Randomly generates the first generation. The living organisms
 * are represented by a 1, and the dead organisms by a 0.
 ****************************************************************/
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

/****************************************************************
 * Prints the entire grid to the terminal.
 ****************************************************************/
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

/*************************************************************************************
 * Produces the next generation. It checks the contents of current_generation_copy,
 * calculates the results, and stores them in current_generation. The living organisms
 * are represented by a 1, and the dead organisms by a 0.
 *************************************************************************************/
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
}

void inline swap(char (**a)[SIZE], char (**b)[SIZE])
{
    char (*temp)[SIZE] = *a;
    *a = *b;
    *b = temp;
}
