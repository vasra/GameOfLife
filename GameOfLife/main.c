#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 10

void initial_state(char first_generation[SIZE][SIZE], char first_generation_copy[SIZE][SIZE]);
void print_grid(char life[SIZE][SIZE]);
void next_generation(char current_generation[SIZE][SIZE], char current_generation_copy[SIZE][SIZE]);

int main()
{
    char life[SIZE][SIZE];
    char life_copy[SIZE][SIZE];
    char (*current_generation)[SIZE] = life;
    char (*current_generation_copy)[SIZE] = life_copy;
    char (*temp)[SIZE];

    srand(time(NULL));
    initial_state(current_generation, current_generation_copy);

    printf("------------------------------------\n");
    printf("    Welcome to the Game Of Life!\n");
    printf("------------------------------------\n\n");

    print_grid(current_generation);
    for(int i = 0; i < 4; i++)
    {
        next_generation(current_generation, current_generation_copy);
        print_grid(current_generation);

        /*
         * Swap the addresses of the two tables. That way, we avoid copying the contents
         * of current_generation to current_generation_copy in the next_generation function.
         * Each round, the addresses are exchanged, saving time from running a loop to copy the contents
         */
        temp = current_generation;
        current_generation = current_generation_copy;
        current_generation_copy = temp;
    }
    return 0;
}

void initial_state(char first_generation[SIZE][SIZE], char first_generation_copy[SIZE][SIZE])
{
    float probability;

    for(int i = 0; i < SIZE; i++)
    {
        for(int j = 0; j < SIZE; j++)
        {
            probability = (float)rand() / (float)((unsigned)RAND_MAX + 1);
            if(probability >= 0.5f)
                first_generation[i][j] = first_generation_copy[i][j] = 'X';
            else
                first_generation[i][j] = first_generation_copy[i][j] = '-';
        }
    }
}

void print_grid(char life[SIZE][SIZE])
{
    for (int i = 0; i< SIZE; i++)
    {
        for(int j = 0; j < SIZE; j++)
        {
            printf("%c ", life[i][j]);
            if( j == SIZE - 1)
                printf("\n");
        }
    }
    printf("\n");
}

void next_generation(char current_generation[SIZE][SIZE], char current_generation_copy[SIZE][SIZE])
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
                        if(current_generation_copy[k][l] == 'X')
                            neighbors++;
                    }
                }
            }
            if(current_generation_copy[i][j] == 'X')
                neighbors--;

            if(neighbors == 3 || (neighbors == 2 && current_generation_copy[i][j] == 'X'))
                current_generation[i][j] = 'X';
            else
                current_generation[i][j] = '-';
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
