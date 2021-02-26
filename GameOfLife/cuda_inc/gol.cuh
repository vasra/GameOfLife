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

__global__ void nextGen(char* d_life, char* d_life_copy, const int size);
