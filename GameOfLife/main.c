#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"

#define SIZE 16
#define NDIMS 2

void Initial_state(int rows, int columns, char *first_generation, char *first_generation_copy);
void Print_grid(int rows, int columns, char *life);
void Next_generation(int rows, int columns, char *life, char *life_copy);
void Swap(char **a, char **b);

int main()
{
    srand(time(NULL));
    /*******************************************************************************
     * ARRAYS FOR THE CARTESIAN TOPOLOGY
     * dim_size - Array with two elements
     *     dim_size[0]  - How many processes will be in each row
     *     dim_size[1]  - How many processes will be in each column
     *
     * periods          - Array with two elements, for the periodicity of the two dimensions
     *
     * coords           - Array with two elements, holding the coordinates of each process
     * north, east etc. - The ranks of our eight neighbors
     *******************************************************************************/

    int dim_size[NDIMS], periods[NDIMS], coords[NDIMS];
    int north[NDIMS], east[NDIMS], south[NDIMS], west[NDIMS],
        northeast[NDIMS], southeast[NDIMS], southwest[NDIMS], northwest[NDIMS];

    /*******************************************************************************************************
     * VARIABLES FOR THE CARTESIAN TOPOLOGY
     * reorder          - Indicates if MPI can rearrange the processes more efficiently among the processors
     * rank             - Process rank
     * processes        - the total number of processes in the communicator
     * rows             - The number of rows of the local 2D matrix
     * columns          - The number of columns of the local 2D matrix
     * cartesian2D      - Our new custom Communicator
     *******************************************************************************************************/

    int reorder, rank, processes, rows, columns;
    MPI_Comm cartesian2D;
    MPI_Status status;

    /**< initialize all dimensions to 0, because MPI_Dims_create will throw an error otherwise */
    dim_size[0] = dim_size[1] = 0;

    /**< Our Cartesian topology will be a torus, so both fields of "periods" array will have a value of 1. */
    periods[0] = periods[1] = 1;

    /**< We will allow MPI to efficiently reorder the processes among the different processors. */
    reorder = 1;

    /**< Initialize MPI */
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /**< Let MPI decide which is the best arrangement according to the number of processes and dimensions */
    MPI_Dims_create(processes, NDIMS, dim_size);

    /**< Create a 2D Cartesian topology. Find the rank and coordinates of each process */
    MPI_Cart_create(MPI_COMM_WORLD, NDIMS, dim_size, periods, reorder, &cartesian2D);
    MPI_Cart_coords(cartesian2D, rank, NDIMS, coords);

    /**< Synchronize all the processes before we start */
    MPI_Barrier(cartesian2D);

    //printf("My rank is %d and my coordinates are [%d][%d]\n", rank, coords[0], coords[1]);

    /**< We add 2 to each dimension in order to include the halo rows and columns */
    rows = (SIZE / dim_size[0]) + 2;
    columns = (SIZE /dim_size[1]) + 2;


    /**< Calculate the coordinates of all neighbors */
    north[0] = coords[0] - 1;
    north[1] = coords[1];

    east[0] = coords[0];
    east[1] = coords[1] + 1;

    south[0] = coords[0] + 1;
    south[1] = coords[1];

    west[0] = coords[0];
    west[1] = coords[1] - 1;

    northeast[0] = coords[0] - 1;
    northeast[1] = coords[1] - 1;

    southeast[0] = coords[0] + 1;
    southeast[1] = coords[1] + 1;

    southwest[0] = coords[0] + 1;
    southwest[1] = coords[1] - 1;

    northwest[0] = coords[0] - 1;
    northwest[1] = coords[1] - 1;

    /**< Pointers to our 2D grid, and its necessary copy. */
    char *life = (char*)malloc( rows * columns * sizeof(char) );
    char *life_copy = (char*)malloc( rows * columns * sizeof(char) );

     /**< Generate the first generation according to that */
    Initial_state(rows, columns, life, life_copy);

//    if( rank != 0 )
//    {
//        MPI_Send(life, rows * columns, MPI_CHAR, 0, rank, cartesian2D);
//    }
    if(rank == 0)//else
    {
        printf("The grid for process 0 is:\n");
        Print_grid(rows, columns, life);
        //for(int i = 1; i < processes; i++)
        //{
            //MPI_Recv(life, rows * columns, MPI_CHAR, i, i, cartesian2D, &status);
            //printf("The grid for process %d is:\n", i);
            //Print_grid(rows, columns, life);
            printf("The coordinates and ranks of my neighbors are\n\n");

            MPI_Cart_rank(cartesian2D, north, &rank);
            printf("North [%d, %d] rank %d\n", north[0], north[1], rank);

            MPI_Cart_rank(cartesian2D, northeast, &rank);
            printf("Northeast [%d, %d] rank %d\n", northeast[0], northeast[1], rank);

            MPI_Cart_rank(cartesian2D, east, &rank);
            printf("East [%d, %d] rank %d\n", east[0], east[1], rank);

            MPI_Cart_rank(cartesian2D, southeast, &rank);
            printf("Southeast [%d, %d] rank %d\n", southeast[0], southeast[1], rank);

            MPI_Cart_rank(cartesian2D, south, &rank);
            printf("South [%d, %d] rank %d\n", south[0], south[1], rank);

            MPI_Cart_rank(cartesian2D, southwest, &rank);
            printf("Southwest [%d, %d] rank %d\n", southwest[0], southwest[1], rank);

            MPI_Cart_rank(cartesian2D, west, &rank);
            printf("West [%d, %d] rank %d\n", west[0], west[1], rank);

            MPI_Cart_rank(cartesian2D, northwest, &rank);
            printf("Northwest [%d, %d] rank %d\n", northwest[0], northwest[1], rank);
        //}
    }
    /***********************************************
     * The Game Of Life will run for 5 generations.
     * Modify the number of generations as desired.
     ***********************************************/
//    for(int i = 0; i < 4; i++)
//    {
//        Next_generation(life, life_copy);
//        Print_grid(life);

        /************************************************************************************************
         * Swap the addresses of the two tables. That way, we avoid copying the contents
         * of life to life_copy in the Next_generation function.
         * Each round, the addresses are exchanged, saving time from running a loop to copy the contents.
         ************************************************************************************************/
//        Swap(&life, &life_copy);
//    }

    /**< Clean up and exit */
    free(life);
    free(life_copy);
    MPI_Finalize();
    return 0;
}

/****************************************************************
 * Randomly generates the first generation. The living organisms
 * are represented by a 1, and the dead organisms by a 0.
 ****************************************************************/
void Initial_state(int rows, int columns, char *first_generation, char *first_generation_copy)
{
    float probability;

    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < columns; j++)
        {
            /**< Initialize all halo values to 0. The rest will be assigned values randomly */
            if( i == 0 || j == 0 || i == rows - 1 || j == columns - 1)
            {
                *(first_generation + i * columns + j) = *(first_generation_copy + i * columns + j) = 0;
                continue;
            }
            probability = (float)rand() / (float)((unsigned)RAND_MAX + 1);
            if(probability >= 0.5f)
                *(first_generation + i * columns + j) = *(first_generation_copy + i * columns + j) = 1;
            else
                *(first_generation + i * columns + j) = *(first_generation_copy + i * columns + j) = 0;
        }
    }
}

/****************************************************************
 * Prints the entire grid to the terminal
 ****************************************************************/
void Print_grid(int rows, int columns, char *life)
{
    for (int i = 0; i< rows; i++)
    {
        for(int j = 0; j < columns; j++)
        {
            printf("%d ", *(life + i * columns + j));
            if( j == columns - 1)
                printf("\n");
        }
    }
    printf("\n");
}

/*************************************************************************************
 * Produces the next generation. It checks the contents of life_copy,
 * calculates the results, and stores them in life. The living organisms
 * are represented by a 1, and the dead organisms by a 0.
 *************************************************************************************/
void inline Next_generation(int rows, int columns, char *life, char *life_copy)
{
    int neighbors;
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < columns; j++)
        {
            neighbors = 0;
            for(int k = i - 1; k <= i + 1; k++)
            {
                for(int l = j - 1; l <= j + 1; l++)
                {
                    if(k > -1 && k < rows && l > -1 && l < columns)
                    {
                        if( *(life_copy + k * columns + l) == 1)
                            neighbors++;
                    }
                }
            }
            if( *(life_copy + i * columns + j) == 1)
                neighbors--;

            if(neighbors == 3 || (neighbors == 2 && *(life_copy + i * columns + j) == 1))
                *(life_copy + i * columns + j) = 1;
            else
                *(life_copy + i * columns + j) = 0;
        }
    }
}

void inline Swap(char **a, char **b)
{
    char *temp = *a;
    *a = *b;
    *b = temp;
}
