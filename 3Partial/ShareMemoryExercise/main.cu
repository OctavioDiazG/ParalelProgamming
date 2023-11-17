/*
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





/*
// CUDA Kernel to calculate the sum of each column in a matrix
__global__ void columnSum(float *d_M, float *d_V, int width, int height) {
    int tx = threadIdx.x; // thread index
    int bx = blockIdx.x; //Block index
    int Row = bx; // Row index

    // Ensure the row is within the height of the matrix
    if(Row < height){
        float sum = 0.0;
        // sum up the elements in the column
        for(int i = 0; i < width; i++){
            sum += d_M[Row * width + i];
        }
        // Store the sum in the vector
        d_V[Row] = sum;
    }
}

int main(int argc, char *argv[]) {
    // Parse command line arguments
    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    int padding = atoi(argv[3]);

    // Allocate memory for the matrix and vector
    float *h_M = (float *)malloc(sizeof(float) * (width + 2 * padding) * (height + 2 * padding));
    float *h_V = (float *)malloc(sizeof(float) * (height + 2 * padding));

    // Initialize the matrix with random values
    for(int i = 0; i < (width + 2 * padding) * (height + 2 * padding); i++){
        h_M[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory for the matrix and vector
    float *d_M, *d_V;
    cudaMalloc((void **) &d_M, sizeof(float) * (width + 2 * padding) * (height + 2 * padding));
    cudaMalloc((void **) &d_V, sizeof(float) * (height + 2 * padding));

    // Copy the matrix from host to device memory
    cudaMemcpy(d_M, h_M, sizeof(float) * (width + 2 * padding) * (height + 2 * padding), cudaMemcpyHostToDevice);

    // define the block and grid dimensions
    dim3 dimBlock(BLOCK_SIZE, 1);
    dim3 dimGrid(height / dimBlock.x, 1);

    // Launch the kernel
    columnSum<<<dimGrid, dimBlock>>>(d_M, d_V, width + 2 * padding, height + 2 * padding);

    // Copy the vector from device to host memory
    cudaMemcpy(h_V, d_V, sizeof(float) * (height + 2 * padding), cudaMemcpyDeviceToHost);

    // print the sum of each column
    for(int i = 0; i < height + 2 * padding; i++){
        printf("La suma de la columna %d es: %f\n", i, h_V[i]);
    }

    // Free the host and the dice memory
    free(h_M);
    free(h_V);
    cudaFree(d_M);
    cudaFree(d_V);

    return 0;
}
*/



/*
// CUDA kernel to calculate the sum of each column in a matrix
__global__ void columnSum(float *d_M, float *d_V, int width, int height) {
    __shared__ float sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = 0.0f;
    while (i < width * height) {
        sdata[tid] += d_M[i];
        i += blockDim.x * gridDim.x;
    }
    __syncthreads();
    if (tid == 0) {
        for (i = 1; i < blockDim.x; i++)
            sdata[0] += sdata[i];
        d_V[blockIdx.x] = sdata[0];
    }
}

int main(int argc, char *argv[]) {
    // Parse command line arguments
    int width = atoi(argv[1]);
    int height = atoi(argv[2]);

    // Allocate host memory for the matrix and vector
    float *h_M, *h_V;
    cudaMallocHost((void **) &h_M, sizeof(float) * width * height);
    cudaMallocHost((void **) &h_V, sizeof(float) * height);

    // Initialize the matrix with random values
    for(int i = 0; i < width * height; i++){
        h_M[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory for the matrix and vector
    float *d_M, *d_V;
    cudaMalloc((void **) &d_M, sizeof(float) * width * height);
    cudaMalloc((void **) &d_V, sizeof(float) * height);

    // Copy the matrix from host to device memory
    cudaMemcpy(d_M, h_M, sizeof(float) * width * height, cudaMemcpyHostToDevice);

    // Define the block and grid dimensions
    dim3 dimBlock(BLOCK_SIZE, 1);
    dim3 dimGrid(height / dimBlock.x, 1);

    // Launch the kernel
    columnSum<<<dimGrid, dimBlock>>>(d_M, d_V, width, height);

    // Copy the vector from device to host memory
    cudaMemcpy(h_V, d_V, sizeof(float) * height, cudaMemcpyDeviceToHost);

    // Print the sum of each column
    for(int i = 0; i < height; i++){
        printf("The sum of column %d is: %f\n", i, h_V[i]);
    }

    // Free the host and device memory
    cudaFreeHost(h_M);
    cudaFreeHost(h_V);
    cudaFree(d_M);
    cudaFree(d_V);

    return 0;
}
*/