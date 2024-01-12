# Resume Of Data Parallel Computing Chptr 2

## Chapter outline

1 Data Parallelism <br>
2 CUDA C Program Structure <br>
3 A Vector Addition Kernel  <br>
4 Device Global Memory and Data Transfer <br>
5 Kernel Functions and Threading  <br>
6 Kernel Launch <br>
7 Summary <br>
 7.1 Function Declarations  <br>
 7.2 Kernel Launch  <br>
 7.3 Built-in (Predefined) Variables  <br>
 7.4 Run-time API  <br>
8 Exercises <br>

## Brief Introduction

**Data Parallelism:** Data parallelism is a concept in parallel computing that involves organizing and expressing massively parallel computation based on the data being processed. It is particularly useful when dealing with large amounts of data, such as manipulating images or videos with millions to trillions of pixels, modeling fluid dynamics using billions of grid cells, or simulating interactions between thousands to millions of atoms. The idea behind data parallelism is to break down the computation into smaller independent tasks that can be executed in parallel, resulting in faster overall processing. This approach allows for efficient utilization of massively parallel processors and can significantly improve application

**CUDA C Program Structure:** In CUDA C programming, the structure of a program reflects the coexistence of a host (CPU) and one or more devices (GPUs) in the computer. By default, any traditional C program is a CUDA program that contains only host code. However, one can add device functions and data declarations into any source file.

The CUDA C program structure allows for the inclusion of both host and device code in the same source file. Device functions and data declarations are marked with special CUDA C keywords, indicating that they exhibit a rich amount of data parallelism.

This structure enables the programmer to take advantage of the parallel processing capabilities of GPUs by offloading specific computations to the device. The host code is executed on the CPU, while the device code is executed on the GPU. The CUDA compiler, NVCC, processes the CUDA C program, recognizing and understanding the additional declarations, and compiles the code accordingly.

Overall, the CUDA C program structure facilitates the utilization of data parallelism for faster execution by leveraging the power of GPUs alongside traditional CPU processing.

**A Vector Addition Kernel:** In this section, the document introduces a vector addition kernel as an example of a simple data parallel computation. The kernel code for vector addition is shown, which uses a for-loop to iterate through the vector elements and perform the addition operation. The length of the vectors is controlled by a parameter, and the kernel function is called from a host function to generate a grid of threads on a device. This vector addition example serves as a basic illustration of the CUDA C program structure and the concept of data parallel computing

**Device Global Memory and Data Transfer:** In current CUDA systems, devices often come with their own dynamic random access memory (DRAM) called global memory. This global memory is used to execute a kernel on a device. To do this, the programmer needs to allocate global memory on the device and transfer relevant data from the host memory to the allocated device memory. After the device execution, the programmer needs to transfer the result data from the device memory back to the host memory and free up the device memory that is no longer needed. The CUDA run-time system provides API functions to perform these activities on behalf of the programmer.

**Kernel Functions and Threading:** Kernel functions in CUDA C are used to initiate parallel execution by launching a large number of threads to process different parts of the data in parallel. Each thread has its own code, execution point, and variables. Built-in variables like threadIdx, blockDim, and blockIdx are used by threads to distinguish themselves and determine the area of data each thread should work on.

**Kernel Launch:** Kernel launch is the process of initiating parallel execution by launching a kernel function. The number of threads and blocks to be generated can be specified when launching the kernel. The threads and blocks are organized in a grid structure, and each thread is assigned a unique thread index and block index. The grid structure allows for efficient data parallelism and exploitation of GPU hardware.

**Summary:** This chapter provided an overview of the CUDA C programming model, which extends the C language to support parallel computing. It introduced key concepts such as function declarations, built-in variables, and runtime API functions. The chapter also discussed the structure of a CUDA C program, including the use of kernel functions and device global memory. The goal was to introduce the core concepts of CUDA C and its extensions for writing parallel programs.

## Detailed Breakdown

**Data Parallelism:** 
Task parallelism and data parallelism are two types of parallelism used in parallel programming. Task parallelism involves decomposing an application into independent tasks that can be executed concurrently. For example, a simple application may have tasks like vector addition and matrix-vector multiplication. Data parallelism, on the other hand, involves dividing a large data set into smaller chunks and processing them in parallel. It is the main source of scalability for parallel programs, as it allows for the utilization of massively parallel processors. While data parallelism is more commonly used, task parallelism can also be important in achieving performance goals, especially in large applications with a larger number.

In an RGB representation, each pixel in an image is stored as a tuple of (r, g, b) values. The values of r, g, and b represent the intensity of the red, green, and blue light sources when the pixel is rendered. The intensity values range from 0 (dark) to 1 (full intensity). The format of an image's row is (r g b) (r g b) (r g b), with each tuple specifying a mixture of red, green, and blue. The actual combinations of these colors can vary depending on the color space being used. In the AdobeRGB color space, the valid combinations are shown as the interior of a triangle, with the mixture coefficients (x, y, 1 - y - x) indicating the fraction of pixel intensity assigned to each color.

To convert a color image to greyscale, the luminance value (L) for each pixel is computed using a weighted sum formula. The formula is L = 0.21 * r + 0.72 * g + 0.07 * b, where r, g, and b represent the red, green, and blue values of the pixel, respectively.

Computation Structure

The input image is organized as an array (I) of RGB values, and the output is a corresponding array (O) of luminance values. The computation structure is simple, where each output pixel (O[i]) is generated by calculating the weighted sum of the RGB values in the corresponding input pixel (I[i]). These computations are performed independently for each pixel, without any dependencies between them.

Data Parallelism

The color-to-greyscale conversion process exhibits a rich amount of data parallelism. Each pixel's computation can be performed independently of others, allowing for parallel execution. This parallelism can significantly speed up the overall conversion process. However, it's important to note that data parallelism in complete applications can be more complex, and this book focuses on teaching the necessary "parallel thinking" to identify and exploit data parallelism effectively.

**CUDA C Program Structure:**
once device functions and data declarations are added to a source file, it cannot be compiled by a traditional C compiler. Instead, it needs to be compiled by a CUDA C compiler called NVCC. The NVCC compiler processes the CUDA C program by using CUDA keywords to separate the host code (ANSI C code) and the device code. The host code is compiled with the host's standard C/C++ compilers and runs as a traditional CPU process. The device code, marked with CUDA keywords, is compiled by the NVCC runtime component and executed on a GPU device. In some cases, if there is no hardware device available or a kernel can be appropriately executed on a CPU, the kernel can also be executed on a CPU using tools like MCUDA. The execution of a CUDA program starts with the host code and when a kernel function is called, it is executed by multiple threads on a device. These threads are collectively called a grid and are the primary vehicle of parallel execution in a CUDA platform. After the kernel execution, the grid terminates and the execution continues on the host until another kernel is launched. It is important to note that the given figure (Fig. 2.4) shows a simplified model where the CPU and GPU execution do not overlap, but in reality, many heterogeneous computing applications manage overlapped CPU and GPU execution to take advantage of both CPUs and GPUs.

Explanation in short paragraphs:

- Once device functions and data declarations are added to a source file, it cannot be compiled by a traditional C compiler. <br>
- The code needs to be compiled by a CUDA C compiler called NVCC, which recognizes and understands these additional declarations. <br>
- The NVCC compiler processes a CUDA C program using CUDA keywords to separate the host code (ANSI C code) and the device code.<br> 
- The host code is compiled with the host's standard C/C++ compilers and runs as a traditional CPU process. <br>
- The device code, marked with CUDA keywords, is compiled by the NVCC runtime component and executed on a GPU device. <br>
- In some cases, if there is no hardware device available or a kernel can be appropriately executed on a CPU, the kernel can also be executed on a CPU using tools like MCUDA. <br>
- The execution of a CUDA program starts with the host code and when a kernel function is called, it is executed by multiple threads on a device. <br>
- These threads are collectively called a grid and are the primary vehicle of parallel execution in a CUDA platform. <br>
- After the kernel execution, the grid terminates and the execution continues on the host until another kernel is launched. <br>
- In reality, many heterogeneous computing applications manage overlapped CPU and GPU execution to take advantage of both CPUs and GPUs. <br>

Launching a kernel typically generates a large number of threads to exploit data parallelism. This means that when a kernel is launched in CUDA, it creates multiple threads that can process different parts of the data in parallel.

In the color-to-greyscale conversion example, each thread could be used to compute one pixel of the output array O. This means that in the specific example of converting a color image to grayscale, each thread can be assigned to calculate the value of one pixel in the output array.

The number of threads that will be generated by the kernel is equal to the number of pixels in the image. This means that the total number of threads created by the kernel will be equal to the number of pixels in the image being processed.

For large images, a large number of threads will be generated. This means that if the image being processed is large, a significant number of threads will be created to handle the computation.

In practice, each thread may process multiple pixels for efficiency. This means that to improve efficiency, each thread may be assigned to calculate the values of multiple pixels instead of just one.

CUDA programmers can assume that these threads take very few clock cycles to generate and schedule due to efficient hardware support. This means that CUDA programmers can expect the threads to be created and scheduled quickly, as the hardware is designed to efficiently handle the generation and scheduling of threads.

This is in contrast with traditional CPU threads that typically take thousands of clock cycles to generate and schedule. This means that in comparison to traditional CPU threads, which can take a long time to generate and schedule, the CUDA threads are much more efficient in terms of time required for these operations.

**A Vector Addition Kernel:**

Vector addition is considered the simplest form of data parallel computation, similar to the "Hello World" program in sequential programming. The text also mentions that the host code, which is the traditional C program, consists of a main function and a vector addition function. In the examples provided, variables processed by the host are prefixed with "h_" to distinguish them from variables processed by the device. The vecAdd function uses a for-loop to iterate through the vector elements, where each iteration calculates the sum of corresponding elements from two input vectors and stores the result in the output vector. The length of the vectors is controlled by the vector length parameter, and the function uses pass-by-reference to read the input elements and write.

The given text explains the structure and functionality of a modified
vecAdd
function for executing vector addition in parallel.

When the
```vecAdd```
function returns, the subsequent statements in the main function can access the new contents of vector C.
To execute vector addition in parallel, the
```vecAdd```
function is modified and its calculations are moved to a device (GPU).
The modified
```vecAdd```
function, shown in Figure 2.6, includes three parts:
Part 1: Allocates space in the device memory to hold copies of vectors A, B, and C, and copies the vectors from the host memory to the device memory.
Part 2: Launches parallel execution of the actual vector addition kernel on the device.
Part 3: Copies the sum vector C from the device memory back to the host memory and frees the vectors in the device memory.
In summary, the modified
```vecAdd```
function allows for parallel execution of vector addition by allocating device memory, transferring data between host and device memory, and performing the vector addition calculation on the device.

## Reflexion
### Chapter 3: CUDA Execution Model

In Chapter 3, the focus is on developing kernels with a profile-driven approach and understanding the nature of warp execution. It explains how to expose more parallelism to the GPU and master grid and block configuration heuristics. The chapter also covers various CUDA performance metrics and events, as well as dynamic parallelism and nested execution.

The chapter provides guidelines for selecting grid and block configurations, helping to optimize kernel performance. It emphasizes the importance of understanding hardware resources and the GPU architecture to write better code and fully exploit the device's capability. The chapter also delves into the concept of warp execution and how threads are grouped into warps.

### Chapter 4: Global Memory

Chapter 4 focuses on the CUDA memory model and programming with global memory. It explores global memory access patterns and the impact of memory efficiency on kernel performance. The chapter provides guidelines for improving bandwidth utilization and maximizing the utilization of bytes traveling between global memory and on-chip memory.

The chapter discusses the importance of aligned and coalesced memory accesses and provides techniques for achieving them. It also introduces Unified Memory, which simplifies CUDA programming by eliminating duplicate pointers and the need for explicit data transfer between the host and device.

### Opinion and Conclusion

In my opinion, Chapters 3 and 4 provide valuable insights into optimizing CUDA kernel performance. 

Chapter 3 highlights the importance of understanding the hardware perspective and provides guidelines for selecting grid and block configurations. It also sheds light on the nature of warp execution and how it impacts kernel design.

Chapter 4 delves into the intricacies of global memory and its impact on kernel performance. It emphasizes the importance of memory efficiency, aligned and coalesced memory accesses, and maximizing bandwidth utilization. The introduction of Unified Memory simplifies programming and eliminates the need for explicit data transfer.

Parallelism and Performance: The execution model allows threads to be executed in warps, maximizing parallelism and performance. This SIMT (Single Instruction, Multiple Threads) fashion enables efficient utilization of hardware resources, leading to faster execution of kernels.

Hardware Perspective: The execution model exposes the hardware perspective, allowing developers to understand the underlying architecture and optimize their code accordingly. By considering hardware resources, cache characteristics, and memory access patterns, developers can write more efficient code and fully exploit the capabilities of the GPU.

Profile-Driven Approach: CUDA programming encourages a profile-driven approach, where developers analyze the performance of their code using profiling tools like nvprof. This approach helps identify performance bottlenecks, understand kernel behavior, and guide optimization efforts, resulting in improved performance.

Dynamic Parallelism: The introduction of dynamic parallelism in CUDA enables the creation of new work directly from the device. This feature allows for the expression of recursive or data-dependent parallel algorithms in a more natural and easy-to-understand way, enhancing code flexibility and maintainability.

Overall, By understanding the CUDA execution model and global memory concepts, I can apply optimization techniques to improve performance in both code and real-life applications. Whether it's optimizing algorithms, improving data access patterns, or maximizing resource utilization, the knowledge gained from these chapters can be valuable in various domains where optimization is key.


## References 

[1] J. Cheng, M. Grossman, and T. McKercher, Professional Cuda C Programming. Indianapolis (Ind.): Wrox, 2014. 
