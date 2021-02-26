#include <life.h>

//#define DEBUG

///////////////////////////////////////////////////////////////////////
// size        - The size of one side of the square grid
// generations - The number of generations for which the game will run
// nthreads    - The number of threads per block
// nblocks     - The number of blocks
///////////////////////////////////////////////////////////////////////
#ifndef DEBUG
constexpr int size = 840;
constexpr int generations = 2000;
constexpr int nthreads = 64;
constexpr int nblocks = size * size / nthreads;
const     int blocksPerRow = size / nthreads;
//dim3 dimBl(blocksPerSide, blocksPerSide);
#else
constexpr int size = 8;
constexpr int generations = 2;
constexpr int nthreads = 4;
constexpr int nblocks = size * size / nthreads;
const     int blockSide = static_cast<int>(size / sqrt(nblocks));
dim3 dimBl(blockSide, blockSide);
#endif

int main() {
    
    // Pointers to our 2D grid, and its necessary copy
    // We add 2 to each side, in order to include the halo rows and columns
    char *h_life = (char*)malloc((size + 2) * (size + 2) * sizeof(char) );
    char *h_life_copy = (char*)malloc((size + 2)* (size + 2) * sizeof(char) );

    // Produce the first generation randomly
    Initial_state(size, h_life, h_life_copy);
    
    float msecs = GameOfLife(size, h_life, h_life_copy, nblocks, generations);

    printf("Elapsed time is %.2f msecs\n", msecs);

    // Clean up and exit
    free(h_life);
    free(h_life_copy);
 
    return 0;
}

/////////////////////////////////////////////////////////////////
// Randomly produces the first generation. The living organisms
// are represented by a 1, and the dead organisms by a 0.
/////////////////////////////////////////////////////////////////
void Initial_state(int size, char * h_life, char * h_life_copy) {
    float randomProbability = 0.0f;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> probability(0.0f, 1.0f);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            // Initialize all halo values to 0. The rest will be assigned values randomly.
            if (i == 0 || i == size - 1 || j == 0 || j == size - 1) {
                *(h_life + i * size + j) = *(h_life_copy + i * size + j) = 0;
            } else {
                randomProbability = static_cast<float>(probability(gen));
                if (randomProbability >= 0.5f)
                    *(h_life + i * size + j) = *(h_life_copy + i * size + j) = 1;
                else
                    *(h_life + i * size + j) = *(h_life_copy + i * size + j) = 0;
            }
        }
    }
}

/////////////////////////////////////////////////////////////////
// Prints the entire grid to the terminal. Used for debugging
/////////////////////////////////////////////////////////////////
void Print_grid(int size, char * h_life) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%d ", *(h_life + i * size + j));
            if ( j == size - 1)
                printf("\n");
        }
    }
    printf("\n");
}

//////////////////////////////////////////////////////////////////////////////////////
// Plays the Game Of Life. It checks the contents of d_life,
// calculates the results, and stores them in d_life_copy. The living organisms
// are represented by a 1, and the dead organisms by a 0.
//////////////////////////////////////////////////////////////////////////////////////
float GameOfLife(const int size, char* h_life, char* h_life_copy, int nblocks, dim3 dimBl, int generations) {
    // The grids that will be copied to the GPU
    char* d_life;
    char* d_life_copy;
    cudaError_t err;

    err = cudaMalloc((void**)&d_life, size * size * sizeof(char));
    if (err != cudaSuccess) {
        fprintf(stderr, "Could not allocate CUDA memory, with error code %d\n", err);
        return err;
    }

    err = cudaMalloc((void**)&d_life_copy, size * size * sizeof(char));
    if (err != cudaSuccess) {
        fprintf(stderr, "Could not allocate CUDA memory, with error code %d\n", err);
        return err;
    }

    err = cudaMemcpy(d_life, h_life, sizeof(char) * size * size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Could not copy to GPU memory, with error code %d\n", err);
        return err;
    }

    err = cudaMemcpy(d_life_copy, h_life_copy, sizeof(char) * size * size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Could not copy to GPU memory, with error code %d\n", err);
        return err;
    }

    timestamp t_start;
    t_start = getTimestamp();

    for (int gen = 0; gen < generations; gen++) {
        nextGen << <nblocks, dimBl >> > (d_life, d_life_copy, size, nblocks, dimBl, generations);

        /////////////////////////////////////////////////////////////////////////////////////////////////
         // Swap the addresses of the two tables. That way, we avoid copying the contents
         // of d_life to d_life_copy. Each round, the addresses are exchanged, saving time from running
         // a loop to copy the contents.
         /////////////////////////////////////////////////////////////////////////////////////////////////
        std::swap(d_life, d_life_copy);
    }

#ifdef DEBUG
    cudaMemcpy(h_life, d_life, sizeof(char) * size * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_life_copy, d_life_copy, sizeof(char) * size * size, cudaMemcpyDeviceToHost);
    printf("Generation %d\n", gen);
    printf("life\n");
    Print_grid(size, h_life);
    printf("life_copy\n");
    Print_grid(size, h_life_copy);
#endif

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error synchronizing devices: %s\n", err);
        return err;
    }

    float msecs = getElapsedtime(t_start);

    err = cudaFree(d_life);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error freeing GPU memory: %s\n", err);
        return err;
    }

    err = cudaFree(d_life_copy);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error freeing GPU memory: %s\n", err);
        return err;
    }

    return msecs;
}