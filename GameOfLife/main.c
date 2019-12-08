#include <stdio.h>
#include <stdlib.h>

#define SIZE 16

void dead_state(int width, int height, char life_grid[SIZE][SIZE]);
void print_grid(char life_grid[SIZE][SIZE]);

int main()
{
    char life_grid[SIZE][SIZE];

    for(int i = 0; i < SIZE; i++)
    {
        for(int j = 0; j < SIZE; j++)
            life_grid[i][j] = 'X';
    }
    print_grid(life_grid);
    return 0;
}

void dead_state(int width, int height, char life_grid[SIZE][SIZE])
{
    life_grid[width][height] = '-';
}

void print_grid(char life_grid[SIZE][SIZE])
{
    printf("------------------------------------\n");
    printf("    Welcome to the Game Of Life!\n");
    printf("------------------------------------\n\n");

    for (int i = 0; i< SIZE; i++)
    {
        for(int j = 0; j < SIZE; j++)
        {
            printf("%c ", life_grid[i][j]);
            if( j == SIZE - 1)
                printf("\n");
        }
    }
}
