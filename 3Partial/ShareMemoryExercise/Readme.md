# Shared Memory Exercise

```C++
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Define the dimensions of the matrix and the block size for the CUDA kernel
#define X 5
#define Y 5
#define PADDING_SIZE_X 1
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

// CUDA kernel function to apply padding to the matrix and sum the columns
__global__ void applyPaddingAndSumColumns(int *matrix, int *paddedMatrix, int *result, int width, int height, int paddingX) {
    // Calculate the column and row index for the current thread
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the bounds of the matrix
    if (col < width && row  < height) {
        // Apply padding to the matrix
        paddedMatrix[row * (width + paddingX) + col + paddingX] = matrix[row * width + col];
        // Sum the columns of the matrix using atomic addition to avoid race conditions
        atomicAdd(&result[col], matrix[row * width + col]);
    }
}

int main() {
    // Declare and initialize the matrix, the padded matrix, and the result array
    int matrix[X][Y];
    int paddedMatrix[X + PADDING_SIZE_X][Y];
    int result[Y] = {0};

    // Fill the matrix with random numbers between 1 and 9
    for (int i = 0; i < X; ++i) {
        for (int j = 0; j < Y; ++j) {
            matrix[i][j] = (rand() % 9)+1;
        }
    }

    // Print the original matrix
    printf("Original:\n");
    for (int i = 0; i < X; ++i) {
        for (int j = 0; j < Y; ++j) {
            printf("%d\t", matrix[i][j]);
        }
        printf("\n");
    }

    // Declare pointers for the device memory
    int *d_matrix, *d_paddedMatrix, *d_result;

    // Allocate memory on the device for the matrix, the padded matrix, and the result array
    cudaMalloc((void **)&d_matrix, X * Y * sizeof(int));
    cudaMalloc((void **)&d_paddedMatrix, (X + PADDING_SIZE_X) * Y * sizeof(int));
    cudaMalloc((void **)&d_result, Y * sizeof(int));

    // Copy the matrix from host to device
    cudaMemcpy(d_matrix, matrix, X * Y * sizeof(int), cudaMemcpyHostToDevice);

    // Define the dimensions of the grid and blocks for the CUDA kernel
    dim3 gridDim((Y + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (X + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, 1);
    dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);

    // Launch the CUDA kernel
    applyPaddingAndSumColumns<<<gridDim, blockDim>>>(d_matrix, d_paddedMatrix, d_result, Y, X, PADDING_SIZE_X);

    // Copy the padded matrix from device to host
    cudaMemcpy(paddedMatrix, d_paddedMatrix, (X + PADDING_SIZE_X) * Y * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the padded matrix
    printf("\nWith Padding:\n");
    for (int i = 0; i < X + PADDING_SIZE_X; ++i) {
        for (int j = 0; j < Y ; ++j) {
            printf("%d\t", paddedMatrix[i][j]);
        }
        printf("\n");
    }

    // Copy the result array from device to host
    cudaMemcpy(result, d_result, Y * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the column sums
    printf("\nColumn Sums:\n");
    for (int j = 0; j < Y; ++j) {
        printf("%d\t", result[j]);
    }
    printf("\n");

    // Free the device memory
    cudaFree(d_matrix);
    cudaFree(d_paddedMatrix);
    cudaFree(d_result);

    return 0;
}

```

This code performs two main operations on a 2D matrix:

Padding: The ```applyPaddingAndSumColumns``` kernel function applies padding to the input matrix. It adds a specified number of rows (```PADDING_SIZE_X```) filled with zeros at the top of the matrix. The padded matrix is then printed out.

Column Summation: The ```applyPaddingAndSumColumns``` kernel function also calculates the sum of each column in the padded matrix and stores the results in an array. The sums are then printed out.

The ```main``` function initializes the matrices and arrays, allocates memory on the GPU, copies data between the host (CPU) and device (GPU), launches the kernel function, and finally frees the allocated memory.

The program uses a 2D grid of thread blocks, and each block contains a 2D array of threads. The number of blocks and threads is defined by ```BLOCK_SIZE_X``` and ```BLOCK_SIZE_Y```. The grid and block dimensions are set up such that each thread corresponds to one element in the matrix.











