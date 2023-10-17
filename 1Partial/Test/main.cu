#include <iostream>
#include <cuda.h>

int main()
{
    int d;
    cudaDeviceProp deviceProp;



    cudaGetDeviceCount(&d);
    cudaGetDeviceProperties(&deviceProp, 0);


    std::cout<<d << std::endl;
    std::cout << "Name " << deviceProp.name << std::endl;
    std::cout << "Memory " << deviceProp.totalGlobalMem << std::endl;
    std::cout << "ClockRate " << deviceProp.clockRate << std::endl;
    std::cout << "Cuda Cores " << deviceProp.multiProcessorCount << std::endl;
    std::cout << " " << deviceProp.maxThreadsDim[0] << std::endl;
    std::cout << " " << deviceProp.maxThreadsDim[1] << std::endl;
    std::cout << " " << deviceProp.maxThreadsDim[2] << std::endl;
    std::cout << " " << deviceProp.maxGridSize[0] << std::endl;
    std::cout << " " << deviceProp.maxGridSize[1] << std::endl;
    std::cout << " " << deviceProp.maxGridSize[2] << std::endl;
}