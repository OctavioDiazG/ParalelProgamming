//Be advice there are 3 codes in this .cu each one with the respective title

/*
// Teacher code
#include <cuda_runtime.h>
#include <stdio.h>

#define ARRAY_SIZE 128
#define BANK_SIZE 32

__global__ void padArray(int* array) {
// Shared memory with padding
    __shared__ int sharedArray[ARRAY_SIZE + ARRAY_SIZE / BANK_SIZE];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int index = bid * blockDim.x + tid;

// Load data into shared memory with padding
    sharedArray[tid] = array[index];
    __syncthreads();

// Access all keys from the original bank 0 in one clock pulse
    int offset = tid / BANK_SIZE;
    int newIndex = tid + offset;

// Use the modified index for accessing the padded shared memory
    int result = sharedArray[newIndex];

// Print the result for demonstration
    printf("Thread %d: Original Value: %d, Padded Value: %d\n", tid, array[index], result);
}

int main() {
    int array[ARRAY_SIZE];

// Initialize array values (you can replace this with your data)
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        array[i] = i * 2;
    }

    int* d_array;

// Allocate device memory
    cudaMalloc((void**)&d_array, ARRAY_SIZE * sizeof(int));

// Copy array from host to device
    cudaMemcpy(d_array, array, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);

// Define block and grid dimensions
    dim3 blockDim(BANK_SIZE);
    dim3 gridDim((ARRAY_SIZE + blockDim.x - 1) / blockDim.x);

// Launch kernel
    padArray<<<gridDim, blockDim>>>(d_array);

// Synchronize device to ensure print statements are displayed
    cudaDeviceSynchronize();

// Free allocated memory
    cudaFree(d_array);

    return 0;
}
*/

// personal 2nd code
#include <stdio.h>
#include <cuda.h>
#include <curand.h>

#define BLOCK_SIZE 32

__global__ void columnSum(float* array, float* result, int x, int y, int padded_x) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int index = bid * blockDim.x + tid;

    if (index < x) {
        float sum = 0;
        for (int i = 0; i < y; ++i) {
            sum += array[i * padded_x + index];
        }
        result[index] = sum;
    }
}

int main() {
    int x = 128; // Number of columns
    int y = 128; // Number of rows

    // Calculate the padded size
    int padded_x = (x + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    float* h_array = (float*)malloc(padded_x * y * sizeof(float));
    float* h_result = (float*)malloc(x * sizeof(float));

    // Initialize array with random values
    for (int i = 0; i < y; ++i) {
        for (int j = 0; j < x; ++j) {
            h_array[i * padded_x + j] = rand() / (float)RAND_MAX;
        }
    }

    float* d_array;
    float* d_result;

    // Allocate device memory
    cudaMalloc((void**)&d_array, padded_x * y * sizeof(float));
    cudaMalloc((void**)&d_result, x * sizeof(float));

    // Copy array from host to device
    cudaMemcpy(d_array, h_array, padded_x * y * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((x + blockDim.x - 1) / blockDim.x);

    // Launch kernel
    columnSum<<<gridDim, blockDim>>>(d_array, d_result, x, y, padded_x);

    // Copy result from device to host
    cudaMemcpy(h_result, d_result, x * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < x; ++i) {
        printf("Column %d: Sum = %f\n", i, h_result[i]);
    }

    // Free allocated memory
    free(h_array);
    free(h_result);
    cudaFree(d_array);
    cudaFree(d_result);

    return 0;
}





/*
// personal 1st code
#include <stdio.h>
#include <cuda.h>
#include <curand.h>

#define BLOCK_SIZE 32

__global__ void columnSum(float* array, float* result, int x, int y) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int index = bid * blockDim.x + tid;

    if (index < x) {
        float sum = 0;
        for (int i = 0; i < y; ++i) {
            sum += array[i * x + index];
        }
        result[index] = sum;
    }
}

int main() {
    int x = 128; // Number of columns
    int y = 128; // Number of rows

    float* h_array = (float*)malloc(x * y * sizeof(float));
    float* h_result = (float*)malloc(x * sizeof(float));

    // Initialize array with random values
    for (int i = 0; i < x * y; ++i) {
        h_array[i] = rand() / (float)RAND_MAX;
    }

    float* d_array;
    float* d_result;

    // Allocate device memory
    cudaMalloc((void**)&d_array, x * y * sizeof(float));
    cudaMalloc((void**)&d_result, x * sizeof(float));

    // Copy array from host to device
    cudaMemcpy(d_array, h_array, x * y * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((x + blockDim.x - 1) / blockDim.x);

    // Launch kernel
    columnSum<<<gridDim, blockDim>>>(d_array, d_result, x, y);

    // Copy result from device to host
    cudaMemcpy(h_result, d_result, x * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < x; ++i) {
        printf("Column %d: Sum = %f\n", i, h_result[i]);
    }

    // Free allocated memory
    free(h_array);
    free(h_result);
    cudaFree(d_array);
    cudaFree(d_result);

    return 0;
}
*/
