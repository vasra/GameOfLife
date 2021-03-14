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

void InitialState(int size, char* first_generation);
float GameOfLife(const int size, char* life, int generations);