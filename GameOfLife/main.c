#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 4

void initial_state(char life_beginning[SIZE][SIZE]);
void print_grid(char life[SIZE][SIZE]);
void next_generation(char life_current_generation[SIZE][SIZE], char life_next_generation[SIZE][SIZE]);

int main()
{
    char life_beginning[SIZE][SIZE];
    char life_next_generation[SIZE][SIZE];
    srand(time(NULL));

    initial_state(life_beginning);
    print_grid(life_beginning);
    next_generation(life_beginning, life_next_generation);
    print_grid(life_next_generation);
    return 0;
}

void initial_state(char life_beginning[SIZE][SIZE])
{
    float probability;

    for(int i = 0; i < SIZE; i++)
    {
        for(int j = 0; j < SIZE; j++)
        {
            probability = (float)rand() / (float)((unsigned)RAND_MAX + 1);
            if(probability >= 0.5f)
                life_beginning[i][j] = 'X';
            else
                life_beginning[i][j] = '-';
        }
    }
}

void print_grid(char life[SIZE][SIZE])
{
    printf("------------------------------------\n");
    printf("    Welcome to the Game Of Life!\n");
    printf("------------------------------------\n\n");

    for (int i = 0; i< SIZE; i++)
    {
        for(int j = 0; j < SIZE; j++)
        {
            printf("%c ", life[i][j]);
            if( j == SIZE - 1)
                printf("\n");
        }
    }
}

void next_generation(char life_current_generation[SIZE][SIZE], char life_next_generation[SIZE][SIZE])
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
                        if(life_current_generation[k][l] == 'X')
                            neighbors++;
                    }
                }
            }
            if(life_current_generation[i][j] == 'X')
                neighbors--;

                printf("Neighbors %d ", neighbors);
            if(neighbors == 3 || (neighbors == 2 && life_current_generation[i][j] == 'X'))
                life_next_generation[i][j] = 'X';
            else
                life_next_generation[i][j] = '-';
        }
        printf("\n");
    }
}
