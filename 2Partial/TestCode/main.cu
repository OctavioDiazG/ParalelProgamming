#include <cuda_runtime.h>
#include <iostream>

// Function to check and handle CUDA errors
#define CHECK_CUDA(call) \
    { \
        const cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " << \
            cudaGetErrorString(error); \
            exit(1); \
        } \
    }

int getDeviceSMCount() {
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount)); // Get number of devices
    if (deviceCount == 0)
    {
        std::cerr << "No CUDA-capable devices detected.";
        exit(1);
    }

    int device;
    for (device = 0; device < deviceCount; ++device)
    {
        cudaDeviceProp deviceProp;
        CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, device)); // Get device properties

        int smCount = deviceProp.multiProcessorCount; // Get number of Streaming Multiprocessors (SM)
        std::cout << "Device " << device << " has " << smCount << " Streaming Multiprocessors.\n";

        return smCount;
    }
    return -1;
}

int main() {
    int numSMs = getDeviceSMCount();

    std::cout << "Total SMs in device: " << numSMs << "\n";

    return 0;
}