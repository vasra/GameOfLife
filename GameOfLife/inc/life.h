#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <random>
#include <lcutil.h>
#include <timestamp.h>
#include "device_launch_parameters.h"

void Initial_state(int size, char* first_generation);
void Print_grid(int size, char* life);
float GameOfLife(const int size, char* life, int nblocks, int generations);