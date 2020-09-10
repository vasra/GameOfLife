#pragma once

void Initial_state(int rows, int columns, char* first_generation, char* first_generation_copy);
void Print_grid(int rows, int columns, char* life);
extern "C" float GameOfLife(const int rows, const int columns, char* life, char* life_copy, int nblocks, int nthreads, int generations);