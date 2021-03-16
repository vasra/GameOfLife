#include <life.h>
#include <gol.cuh>
#include <assert.h>
#include <iostream>
#include <vector>

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
    std::vector<char> firstRealRow(h_life + size + 3, h_life + size * 2);
    std::vector<char> lastRealRow(h_life + size * (size + 2) + 1, h_life + size * (size + 2) + size);

    assert(firstRealRow.size() == size)
    copyHaloRows<<<copyingBlocksRows, threads>>>(d_life, size);

    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess);

    err = cudaMemcpy(h_life, d_life, (size + 2) * (size + 2) * sizeof(char), cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);

    std::vector<char> topHaloRow(h_life + 1, h_life + size);
    std::vector<char> bottomHaloRow(h_life + (size + 2) * (size + 1) + 1, h_life + (size + 2) * (size + 1) + size);

    assert(firstRealRow == bottomHaloRow);
    assert(lastRealRow  == topHaloRow);

    cudaFree(d_life);
    free(h_life);
}

void copyHaloColumnsTest() {
    char* h_life = (char*)malloc((size + 2) * (size + 2) * sizeof(char));
    assert(h_life != NULL);
    InitialState(size, h_life);

    char* d_life;
    cudaError_t err;
    err = cudaMalloc((void**)&d_life, (size + 2) * (size + 2) * sizeof(char));
    assert(err == cudaSuccess);

    err = cudaMemcpy(d_life, h_life, (size + 2) * (size + 2) * sizeof(char), cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);

    int copyingBlocksColumnss = size / threads;
    std::vector<char> firstRealColumn;
    std::vector<char> lastRealColumn;

    // copy bottom-right corner element
    firstRealColumn.push_back(*(h_life + size * (size + 2) + size));

    // copy bottom-left corner element
    lastRealColumn.push_back(*(h_life + size * (size + 2) + 1));

    // copy rest of the elements
    for (int i = 1; i < size + 1; i++) {
        firstRealColumn.push_back(*(h_life + i * (size + 2) + 1));
        lastRealColumn.push_back(*(h_life + i * (size + 2) + size));
    }

    // copy top-right corner element
    firstRealColumn.push_back(*(h_life + size * 2));

    // copy top-left corner element
    lastRealColumn.push_back(*(h_life + size + 3));

    assert(firstRealColumn.size() == size + 2);
    assert(lastRealColumn.size() == size + 2);

    copyHaloRows<<<copyingBlocksColumns, threads>>> (d_life, size);

    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess);

    err = cudaMemcpy(h_life, d_life, (size + 2) * (size + 2) * sizeof(char), cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);

    std::vector<char> leftHaloColumn;
    std::vector<char> rightHaloColumn;

    // copy halo columns
    for (int i = 0; i < size + 2; i++) {
        leftHaloColumn.push_back(*(h_life + i * (size + 2)));
        rightHaloColumn.push_back(*(h_life + i * (size + 2) + size + 1));
    }

    assert(leftHaloColumn.size() == size + 2);
    assert(rightHaloColumn.size() == size + 2);

    assert(firstRealColumn == rightHaloColumn);
    assert(lastRealColumn == leftHaloColumn);

    cudaFree(d_life);
    free(h_life);
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