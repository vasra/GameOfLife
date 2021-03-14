#include <life.h>
#include <gol.cuh>
#include <assert.h>
#include <iostream>

void PrintGrid(int size, char* h_life);
void InitialState(int size, char* h_life);

int size = 840;
int threads = 512;

void main() {
    std::cout << "Hello world test!" << std::endl;
}

void copyHaloRowsTest() {
    char* h_life = (char*)malloc((size + 2) * (size + 2) * sizeof(char));
    assert(h_life != NULL);
    InitialState(size, h_life);

    char* d_life;
    cudaError_t err;
    err = cudaMalloc((void**)&d_life, (size + 2) * (size + 2) * sizeof(char));
    assert(err == cudaSuccess);

    err = cudaMemcpy(d_life, h_life, (size + 2) * (size + 2) * sizeof(char), cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);

    int copyingBlocksRows = size / threads;
    copyHaloRows<<<copyingBlocksRows, threads>>>(d_life, size);

    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess);
}

/////////////////////////////////////////////////////////////////
// Prints the entire grid to the terminal. Used for debugging
/////////////////////////////////////////////////////////////////
void PrintGrid(int size, char* h_life) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%d ", *(h_life + i * size + j));
            if (j == size - 1)
                printf("\n");
        }
    }
    printf("\n");
}

/////////////////////////////////////////////////////////////////
// Randomly produces the first generation. The living organisms
// are represented by a 1, and the dead organisms by a 0.
/////////////////////////////////////////////////////////////////
void InitialState(int size, char* h_life) {
    float randomProbability = 0.0f;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> probability(0.0f, 1.0f);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            // Initialize all halo values to 0. The rest will be assigned values randomly.
            if (i == 0 || i == size - 1 || j == 0 || j == size - 1) {
                *(h_life + i * size + j) = 0;
            } else {
                randomProbability = static_cast<float>(probability(gen));
                if (randomProbability >= 0.5f)
                    *(h_life + i * size + j) = 1;
                else
                    *(h_life + i * size + j) = 0;
            }
        }
    }
}