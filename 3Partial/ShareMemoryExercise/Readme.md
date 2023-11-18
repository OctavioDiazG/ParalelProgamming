# Shared Memory Exercise

```C++
#include <stdio.h>
#include <cuda.h>
#include <curand.h>

#define BLOCK_SIZE 32

__global__ void columnSum(float* array, float* result, int x, int y, int padded_x) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int index = bid * blockDim.x + tid;

    if (index < padded_x) {
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
    float* h_result = (float*)malloc(padded_x * sizeof(float));

    // Initialize array with random values
    for (int i = 0; i < y; ++i) {
        for (int j = 0; j < padded_x; ++j) {
            if (j < x) {
                h_array[i * padded_x + j] = rand() / (float)RAND_MAX;
            } else {
                h_array[i * padded_x + j] = -1; // Initialize padded elements
            }
        }
    }

    float* d_array;
    float* d_result;

    // Allocate device memory
    cudaMalloc((void**)&d_array, padded_x * y * sizeof(float));
    cudaMalloc((void**)&d_result, padded_x * sizeof(float));

    // Copy array from host to device
    cudaMemcpy(d_array, h_array, padded_x * y * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((padded_x + blockDim.x - 1) / blockDim.x);

    // Launch kernel
    columnSum<<<gridDim, blockDim>>>(d_array, d_result, x, y, padded_x);

    // Copy result from device to host
    cudaMemcpy(h_result, d_result, padded_x * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < padded_x; ++i) {
        printf("Column %d: Sum = %f\n", i, h_result[i]);
    }

    // Free allocated memory
    free(h_array);
    free(h_result);
    cudaFree(d_array);
    cudaFree(d_result);

    return 0;
}
```

This CUDA C code creates a 2D array of size X * Y with random values, pads the array to a size of your choice, and then performs the sum of the values of each column in parallel.

The main function initializes a 2D array with random values for the original elements and a specific value (-1) for the padded elements. It then allocates memory on the GPU, copies the array from host to device, and launches the ```columnSum``` kernel. After the kernel execution, it copies the result from device to host, prints the sum of each column (including the padded ones), and then frees the allocated memory.

The ```columnSum``` kernel calculates the sum of each column (including the padded elements) and stores the results in the ```result``` array. Each thread is responsible for summing up the elements of a single column.

The size of the array is padded to the nearest multiple of ```BLOCK_SIZE``` (which is 32 in this case) that is greater than or equal to ```x```. This is done to ensure that each warp operates on a contiguous block of memory, which can improve memory access efficiency.


**Kernel Definition:** (```columnSum```): This is a CUDA kernel that operates on a 2D array of floats. It calculates the sum of each column (including the padded elements) and stores the results in the ```result``` array. Each thread is responsible for summing up the elements of a single column.

**Main Function:** The ```main``` function initializes a 2D array with random values for the original elements and a specific value (-1) for the padded elements. It then allocates memory on the GPU, copies the array from host to device, and launches the ```columnSum``` kernel. After the kernel execution, it copies the result from device to host, prints the sum of each column (including the padded ones), and then frees the allocated memory.

**Padding the Array:** The size of the array is padded to the nearest multiple of ```BLOCK_SIZE``` (which is 32 in this case) that is greater than or equal to ```x```. This is done by the following line of code: 
```C++
int padded_x = (x + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
```

**Initializing the Array:** The array is initialized with random values for the original elements and a specific value (-1) for the padded elements. This is done by the following lines of code:
```C++
for (int i = 0; i < y; ++i) {
    for (int j = 0; j < padded_x; ++j) {
        if (j < x) {
            h_array[i * padded_x + j] = rand() / (float)RAND_MAX;
        } else {
            h_array[i * padded_x + j] = -1; // Initialize padded elements
        }
    }
}
```

**Allocating Device Memory:** Memory is allocated on the GPU for the array and the result using the ```cudaMalloc``` function.

**Copying Array from Host to Device:** The array is copied from host memory to device memory using the ```cudaMemcpy``` function.

**Defining Block and Grid Dimensions:** The block and grid dimensions are defined using the ```dim3``` data type. The block size is set to ```BLOCK_SIZE```, and the grid size is calculated as the ceiling of the ratio of the padded size of the array to the block size.

**Launching the Kernel:** The ```columnSum``` kernel is launched with the defined grid and block dimensions, and with the array, result, ```x```, ```y```, and ```padded_x``` as arguments.

**Copying Result from Device to Host:** After the kernel execution, the result is copied from device memory to host memory using the ```cudaMemcpy``` function.

**Printing the Result:** The sum of each column (including the padded ones) is printed to the console.

**Freeing Allocated Memory:** The allocated host and device memory is freed using the ```free``` and ```cudaFree``` functions, respectively.













