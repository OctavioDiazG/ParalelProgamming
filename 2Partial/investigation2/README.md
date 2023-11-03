# Resume Of Professional CUDA C Programming Chptrs 3-4

## Overall Overview
**Chapter 3:** focuses on the CUDA execution model and the common limiters to kernel performance, such as memory bandwidth, compute resources, and instruction and memory latency. It emphasizes the importance of understanding hardware resources to improve kernel performance.

**Chapter 4:** delves into the topic of global memory in CUDA programming. It explains the CUDA memory model and explores different global memory access patterns. The chapter also covers memory management, memory allocation and deallocation, memory transfer, and performance tuning techniques related to global memory.

## Overall Key points
### Chapter 3:
- The CUDA execution model has three common limiters to kernel performance: memory bandwidth, compute resources, and instruction and memory latency.
- Understanding hardware resources is crucial for optimizing kernel performance.
- Knowledge of the GPU architecture enables better code writing and utilization of device capabilities.

### Chapter 4:
- The CUDA memory model unifies host and device memory systems and allows explicit control over data placement for optimal performance.
- Applications often exhibit temporal and spatial locality, which can be leveraged for efficient memory access.
- The chapter covers various global memory access patterns, including aligned and coalesced access, and discusses performance tuning techniques for maximizing memory bandwidth.

## Chapter 3: Kuda Execution Model
3.1 Introducing the CUDA Execution Model <br>
The CUDA execution model provides an operational view of how instructions are executed on a specific computing architecture. It exposes an abstract view of the GPU parallel architecture, allowing developers to reason about thread concurrency. The execution model consists of two primary abstractions: a memory hierarchy and a thread hierarchy. The memory hierarchy allows for efficient memory accesses, while the thread hierarchy enables control over the massively parallel GPU.

3.2 Understanding The Nature of Warp Execution <br>
In this section, the document explains the concept of warp execution in detail. It starts by introducing the idea of warps, which are groups of 32 threads that are executed together on a single execution unit called a Streaming Multiprocessor (SM). The document emphasizes that while threads within a warp are logically executed in parallel, not all threads can physically execute in parallel at the same time.

3.3 Exposing Parallelism <br>
It explains that dynamic parallelism allows for the creation of new work directly from the GPU, enabling the expression of recursive or data-dependent parallel algorithms in a more natural and easy-to-understand way. The document also mentions that attention must be given to the child grid launch strategy, parent-child synchronization, and the depth of nested levels when implementing an efficient nested kernel. It highlights that the maximum number of kernel nestings will likely be limited due to the device runtime system reserving extra memory at each nesting level. The document emphasizes the importance of synchronization for both performance and correctness and suggests that reducing the number of in-block synchronizations can lead to more efficient nested kernels. It concludes by stating that dynamic parallelism offers the ability to adapt to data-driven decisions or workloads by making launch configuration decisions at runtime on the device.

3.4 Avoid Branch Divergence <br>
Branch divergence refers to the situation where threads within a warp take different code paths. When threads in a warp diverge, the warp serially executes each branch path, disabling threads that do not take that path. This can result in degraded performance as the amount of parallelism within the warp is reduced.

3.5 Unrolling Loops <br>
Loop unrolling is a technique used to improve the performance of loops in CUDA programming. It involves replicating the body of the loop multiple times, reducing the number of iterations and improving instruction throughput. Unrolling loops can be effective for sequential array processing loops where the number of iterations is known prior to execution.

3.6 Dynamic Paralellism <br>
Dynamic Parallelism is a feature introduced in CUDA that allows the GPU to launch new grids dynamically. It enables the GPU to launch nested kernels, eliminating the need for communication with the CPU. With dynamic parallelism, any kernel can launch another kernel and manage any inter-kernel dependencies needed to perform additional work.

## Chapter 4: Global Memory
4.1 Introducing the CUDA memory model <br>
The CUDA memory model is a key aspect of programming with CUDA. It unifies the separate host and device memory systems and exposes the full memory hierarchy, allowing programmers to explicitly control data placement for optimal performance. The memory model provides a way to manage memory access and achieve optimal latency and bandwidth given the hardware memory subsystem.

4.2 Memory Management <br>
The CUDA memory model provides a unified memory hierarchy that allows explicit control over data placement for optimal performance. Memory management plays a crucial role in high-performance computing on modern accelerators. The memory hierarchy consists of separate host and device memory systems, and the CUDA memory model exposes this hierarchy to the programmer.

4.3 Memory Access Patterns <br>
Optimizing memory access patterns is crucial for maximizing global memory throughput and improving kernel performance. Aligned and coalesced memory accesses are preferred, especially for cached loads and global memory writes, as they minimize wasted bandwidth and maximize bus utilization.

4.4 What bandwidth Can a Kernel Achieve <br>
this section provides insights into the factors affecting kernel performance, such as memory latency, memory bandwidth, block size, and different transpose techniques. It emphasizes the importance of optimizing these factors to achieve the best possible bandwidth for a kernel.

4.5 Matrix Addition with unified memory <br>
Unified Memory is a feature introduced in CUDA 6.0 that simplifies memory management in the CUDA programming model. It creates a pool of managed memory that can be accessed by both the CPU and GPU using the same memory address. This eliminates the need for explicit memory copies and allows for easier management of memory allocations.

## Further Chapter 3 Explanation
### 3.1.1 GPU Architectur Overview:
The GPU architecture is built around a scalable array of Streaming Multiprocessors (SM). Each SM is designed to support concurrent execution of hundreds of threads. When a kernel grid is launched, the thread blocks of that kernel grid are distributed among available SMs for execution. The GPU architecture also employs a Single Instruction Multiple Thread (SIMT) architecture to manage and execute threads in groups of 32 called warps. All threads in a warp execute the same instruction at the same time. The Fermi architecture, in particular, features up to 512 accelerator cores called CUDA cores and has six 384-bit GDDR5 DRAM memory interfaces supporting up to a total of 6 GB of global on-board memory.

![GPU Architecture](3.1.1 GPU Architecture.png)

### 3.1.2 The Fermi Architecture:
The Fermi architecture is a GPU computing architecture that was the first to deliver the features required for demanding high-performance computing (HPC) applications. It features up to 512 accelerator cores called CUDA cores, organized into 16 streaming multiprocessors (SM) with 32 CUDA cores each. Fermi also has six 384-bit GDDR5 DRAM memory interfaces supporting up to 6 GB of global on-board memory. It includes a GigaThread engine for distributing thread blocks to the SM warp schedulers. Fermi supports concurrent kernel execution, allowing multiple kernels to be run on the device at the same time.

### 3.1.3 the Kepler Architecture:
- The Kepler GPU architecture, released in the fall of 2012, is a fast and highly efficient, high-performance computing architecture. It introduces several important innovations, including enhanced streaming multiprocessors (SMs), dynamic parallelism, and Hyper-Q.
- The Kepler architecture features 15 streaming multiprocessors (SMs) and six 64-bit memory controllers. It offers improved programmability and power efficiency compared to previous architectures. The SM units in Kepler have several architectural innovations that enhance performance and power efficiency.
- Dynamic parallelism is a key feature introduced with Kepler GPUs. It allows the GPU to dynamically launch new grids and enables any kernel to launch another kernel. This feature simplifies the creation and optimization of recursive and data-dependent execution patterns.
- Hyper-Q is another innovation in the Kepler architecture. It adds more simultaneous hardware connections between the CPU and GPU, allowing CPU cores to run more tasks on the GPU simultaneously. This increases GPU utilization and reduces CPU idle time.
- Overall, the Kepler architecture provides improved performance, power efficiency, and programmability, making it a valuable choice for high-performance computing applications.

![Dynamic Parallelism](3.1.3 Dynamic Parallelism.png) 
<br>
![Fermi vs Kepler](3.1.3 Fermi vs Kepler.png)

### 3.1.4 Profile-Driven Optimization:
Profile-Driven Optimization is the act of analyzing program performance by measuring various factors such as the space or time complexity of application code, the use of particular instructions, and the frequency and duration of function calls. It is a critical step in program development, especially for optimizing HPC (High-Performance Computing) application code. By using profiling tools, developers can identify performance bottlenecks and gain insight into how compute resources are being utilized in CUDA programming. Profiling tools like nvvp and nvprof provide deep insight into kernel performance and help in identifying bottlenecks and guiding optimizations.

### 3.2.1 Warps and thread Blocks:
Warps and thread blocks are fundamental concepts in CUDA execution model. A thread block is a collection of threads organized in a 1D, 2D, or 3D layout. From the hardware perspective, a thread block is a 1D collection of warps. Each thread block consists of multiple warps, and each warp consists of 32 consecutive threads. All threads within a warp execute the same instruction in a Single Instruction Multiple Thread (SIMT) fashion.

- **Thread Block Configuration** <br>
Thread blocks can be configured to be one-, two-, or three-dimensional. However, from the hardware perspective, all threads are arranged one-dimensionally. Each thread has a unique ID in a block, and for a one-dimensional thread block, the unique thread ID is stored in the CUDA built-in variable threadIdx.x. Threads with consecutive values for threadIdx.x are grouped into warps.

- **Warp Divergence** <br>
Warp divergence occurs when threads within a warp take different paths through an application. This can happen when there are conditional branches in the code. If threads in the same warp diverge, the warp serially executes each branch path, disabling threads that do not take that path. Warp divergence can significantly degrade performance as it reduces the amount of parallelism within a warp.

- **Synchronization and Hazards** <br>
Threads within a thread block can share data through shared memory and registers. However, when sharing data between threads, it is important to avoid race conditions or hazards. Race conditions occur when multiple threads access the same memory location in an unordered manner. Proper synchronization techniques, such as using synchronization barriers, can be used to coordinate communication between threads and avoid race conditions.

- **Resource Limits and Occupancy** <br>
There are resource limits imposed on thread blocks, such as the maximum number of threads per block, maximum number of concurrent warps per multiprocessor, and maximum amount of shared memory per multiprocessor. Achieving high occupancy, which refers to the number of concurrent threads or warps per SM, is important for performance optimization. However, full occupancy is not the only goal, and other factors need to be considered for performance tuning.

- **Conclusion** <br>
Understanding the nature of warp execution and thread block configuration is crucial for efficient CUDA programming. It is important to minimize warp divergence, avoid race conditions, and optimize resource usage to achieve high performance.

![warps and Thread blocks](3.2.1 warps and Thread blocks.png)

### 3.2.2 Warp Divergence:
Warp divergence occurs when threads within a warp take different code paths. This can happen when threads in a warp execute different instructions based on conditional statements. When warp divergence occurs, the warp serially executes each branch path, disabling threads that do not take that path. This can result in degraded performance as the amount of parallelism in the warp is reduced. It is important to avoid different execution paths within the same warp to obtain the best performance. <br>
- Warp divergence happens when threads within a warp take different code paths.
- Threads in a warp must execute the same instruction on each cycle.
- If threads of a warp diverge, the warp serially executes each branch path, disabling threads that do not take that path.
- Warp divergence can cause significantly degraded performance as the amount of parallelism in the warp is reduced.
- Different conditional values in different warps do not cause warp divergence.

### 3.2.3 Resource Partitioning:
Resource partitioning is an important consideration in CUDA programming. It involves managing the compute resources available on the GPU to maximize performance. The number of active warps is limited by the compute resources, so it is crucial to be aware of the hardware restrictions and the resources used by the kernel. By maximizing the number of active warps, the GPU utilization can be maximized as well.
- **Partition Camping** <br>
Partition camping is a phenomenon that can occur when accessing global memory. It refers to the situation where memory requests are queued at some partitions while other partitions remain unused. This can lead to suboptimal performance. To improve performance, it is recommended to evenly divide the concurrent access to global memory among partitions. This can be achieved by adjusting the block execution order or using diagonal block coordinate mapping.

- **Guidelines for Improving Bandwidth Utilization** <br>
To improve bandwidth utilization, two guidelines are provided. First, maximize the number of concurrent memory accesses in-flight. This can be done by creating more independent memory requests in each thread or adjusting the grid and block execution configuration. Second, maximize the utilization of bytes that travel on the bus between global memory and on-chip memory. This can be achieved by striving for aligned and coalesced memory accesses.

- **Understanding the Nature of Warp Execution** <br>
When launching a kernel, threads in the kernel run in parallel from a logical point of view. However, not all threads can physically execute in parallel at the same time. Warps are the basic unit of execution in an SM, and threads within a warp are executed in groups of 32. It is important to understand warp execution from the hardware perspective to guide kernel design and optimize performance.

- **CUDA Memory Model** <br>
The CUDA memory model unifies the host and device memory systems and allows explicit control over data placement for optimal performance. It exposes the full memory hierarchy and provides benefits such as improved latency and bandwidth. Applications often exhibit temporal and spatial locality, and the memory hierarchy takes advantage of this by providing progressively lower-latency but lower-capacity memory levels. By understanding the memory model, developers can efficiently use global memory in their kernels.

### 3.2.4 Latency Hiding:
Latency hiding is a crucial concept in CUDA programming that allows GPUs to maximize their utilization and throughput. It involves hiding the latency of instructions by executing other instructions from different resident warps. GPUs are designed to handle a large number of concurrent and lightweight threads, which enables them to hide instruction latency effectively. 

- Latency hiding in CUDA programming is achieved by executing instructions from different resident warps.
- GPUs are designed to handle a large number of concurrent and lightweight threads to maximize throughput.
- Instruction latency can be classified into arithmetic instruction latency and memory instruction latency.
- Arithmetic instruction latency is the time between an arithmetic operation starting and its output being produced.
- Memory instruction latency is the time between a load or store operation being issued and the data arriving at its destination.
- GPUs can hide arithmetic instruction latency more effectively than memory instruction latency.
- The number of active warps per SM plays a crucial role in latency hiding.
- Little's Law can be used to estimate the number of active warps required to hide latency.
- Choosing an optimal execution configuration is important to strike a balance between latency hiding and resource utilization.

### 3.2.5 Occupancy:
Occupancy refers to the ratio of active warps to the maximum number of warps per Streaming Multiprocessor (SM) in a GPU. It is an important metric for optimizing performance in CUDA applications. Achieving high occupancy allows for better utilization of compute resources and can lead to improved performance. However, it is important to note that higher occupancy does not always guarantee higher performance, as other factors can also impact performance.

- Occupancy is the ratio of active warps to the maximum number of warps per SM.
- Higher occupancy allows for better utilization of compute resources.
- Achieving high occupancy can lead to improved performance in CUDA applications.
- However, higher occupancy does not always equate to higher performance, as other factors can also impact performance.

### 3.2.6 Synchronization:
Synchronization in CUDA refers to the coordination of threads in a thread block during the execution of a kernel. It allows threads to wait for each other to reach a specific point in their execution before proceeding further. There are two levels of synchronization in CUDA: system-level and block-level.

**System-level Synchronization**
System-level synchronization involves waiting for all work on both the host and the device to complete. This can be achieved using the 
```CUDA
cudaError_t cudaDeviceSynchronize(void);
```
function, which blocks the host application until all CUDA operations have completed. It ensures that all asynchronous CUDA operations, such as memory copies and kernel launches, have finished before the host continues its execution.

**Block-level Synchronization**
Block-level synchronization, on the other hand, involves waiting for all threads in a thread block to reach the same point in their execution on the device. This is done using the
```CUDA 
__device__void__syncthreads(void);
```
 function, which is called within a kernel. When ``` __syncthreads ``` is called, each thread in the same thread block must wait until all other threads in that block have reached the synchronization point. This ensures that all global and shared memory accesses made by the threads prior to the synchronization point are visible to all other threads in the block after the synchronization.

### 3.2.7 Scalability:
Scalability is a desirable feature in parallel applications as it allows for improved performance by adding additional hardware resources. In the context of CUDA applications, scalability means that running the application on multiple Streaming Multiprocessors (SMs) can halve the execution time compared to running on a single SM. Scalable parallel programs efficiently utilize all compute resources to enhance performance. Scalability is important as it allows for the execution of the same application code on varying numbers of compute cores, known as transparent scalability, reducing the burden on developers and broadening the use-cases for existing applications.

**Scalability in CUDA Programs** <br>
When a CUDA kernel is launched, thread blocks are distributed among multiple SMs, and these blocks can be executed in any order, either in parallel or in series. This independence of execution allows CUDA programs to scale across an arbitrary number of compute cores. For example, a GPU with two SMs can execute two blocks simultaneously, while a GPU with four SMs can execute four blocks at the same time. This scalability is achieved without requiring any code changes, as the execution time of the application scales according to the available resources.

### 3.3.1 Checking Active Warps with nvprof:
Process of checking active warps using nvprof, a profiling tool in CUDA programming. It explains that active warps are the warps that have been allocated compute resources and are ready for execution. The document provides examples of different thread block configurations and their corresponding achieved occupancy values, which indicate the ratio of active warps to the maximum number of warps supported on a streaming multiprocessor (SM). It also highlights that achieving higher occupancy does not always guarantee better performance and that other factors can restrict performance.

### 3.3.2 Checking Memory Operations with nvprof:
Analyze memory operations in CUDA programs. It mentions the use of metrics such as "gld_throughput" and "gld_efficiency" to measure memory read efficiency and global load efficiency, respectively. The text provides examples of using nvprof to check these metrics for different execution configurations of a kernel. It also highlights the importance of load efficiency and how it can impact performance. Additionally, the text mentions the use of nvvp, a visual profiler, to inspect unified memory performance and measure unified memory traffic.

### 3.3.3 Exposing more Parallelism
It explains that dynamic parallelism allows for the creation of new work directly from the GPU, enabling the expression of recursive or data-dependent parallel algorithms in a more natural and easy-to-understand way. By using dynamic parallelism, the decision of how many blocks and grids to create on the GPU can be postponed until runtime, allowing for better utilization of GPU hardware schedulers and load balancers. This can lead to improved performance and adaptability in response to data-driven decisions or workloads. The document also mentions that the ability to create work directly from the GPU reduces the need to transfer execution control and data between the host and device.

### 3.4.1 The parallel Reduction Problem
The parallel reduction problem involves calculating the sum of an array of integers in parallel. Instead of sequentially adding each element, the array can be divided into smaller chunks and each thread can calculate the partial sum for its chunk. The partial results from each chunk are then added together to obtain the final sum. This approach takes advantage of the associative and commutative properties of addition to perform parallel addition efficiently.

![Parallel Reduction Problem](3.4.1 The Parallel Reduction Problem.png)

### 3.4.2 Divergence in Parallel Reduction
Divergence refers to the situation where threads within a warp take different execution paths. This can happen when there is conditional execution within a warp, leading to poor kernel performance. To avoid divergence, techniques such as rearranging data access patterns can be used. One approach is the neighbored pair implementation, where each thread adds two adjacent elements to produce a partial sum. Another approach is the interleaved pair implementation, where paired elements are separated by a given stride. These techniques help reduce or eliminate warp divergence and improve the performance of parallel reduction kernels.

![Divergence in Parallel Reduction](3.4.2 Divergence in Parallel Reduction.png)

### 3.4.3 Improving Divergence in Parallel Reduction
To improve divergence, the array index of each thread can be rearranged to force neighboring threads to perform the addition. This reduces divergence and improves the efficiency of the parallel reduction algorithm. The implementation involves setting the array access index for each thread and using conditional statements to ensure that only certain threads perform the addition. By reducing divergence, the parallel reduction algorithm can achieve better performance.

![Improving Divergence in Parallel Reduction](3.4.3 Improving Divergence in Parallel Reduction.png)

### 3.4.4 Reducing with Interleaved Pairs
Reducing with interleaved pairs involves pairing elements in a given stride. This implementation allows a thread to take two adjacent elements and produce one partial sum at each step. For an array with N elements, this approach requires N - 1 sums and log2N steps. The inputs to a thread in this implementation are strided by half the length of the input on each step. The kernel code for interleaved reduction is provided in the document.

![Reducing with Interleaved Pairs](3.4.4 Reducing with Interleaved Pairs.png)

### 3.5.1 Reducing With Unrolling
Unrolling loops is a technique used to optimize loop execution by reducing the frequency of branches and loop maintenance instructions. It involves writing the body of a loop multiple times instead of using a loop to execute it repeatedly. The number of copies made of the loop body is called the loop unrolling factor. Unrolling loops can improve performance for sequential array processing loops where the number of iterations is known prior to execution of the loop.

In the context of CUDA programming, unrolling loops can be used to improve performance by reducing instruction overheads and creating more independent instructions to schedule. This leads to higher saturation of instruction and memory bandwidth, resulting in more concurrent operations and better performance. Unrolling can be done in different ways, such as unrolling two data blocks by a single thread block or unrolling warps. The choice of unrolling technique depends on the specific requirements of the application.

### 3.5.2 Reducing with Unrolled Warps
Refers to a technique used in CUDA programming to optimize the performance of reduction kernels. It involves unrolling the last few iterations of a reduction loop to avoid executing loop control and thread synchronization logic. By doing so, the compiler can optimize the code and reduce the number of stalls caused by thread synchronization. This technique can lead to improved performance and higher saturation of instruction and memory bandwidth.
### 3.5.3 Reducing with Complete Unrolling
Complete unrolling is a technique used to improve performance in loops where the number of iterations is known at compile-time. By unrolling the loop, the number of times the loop condition is checked is reduced, resulting in fewer instruction overheads. Additionally, the memory operations within each iteration can be issued simultaneously, further improving performance. In CUDA, complete unrolling can be achieved by manually unrolling the loop and performing in-place reduction. This technique can lead to higher saturation of instruction and memory bandwidth, resulting in improved performance.

### 3.5.4 Reducing with Template Functions
Discusses the use of template functions to further reduce branch overhead in parallel reduction. It explains that CUDA supports template parameters on device functions, allowing the block size to be specified as a parameter of the template function. The code example provided demonstrates the use of template functions to implement a parallel reduction kernel with complete loop unrolling. The performance improvement achieved by this approach is also mentioned in the document.

### 3.6.1 Nested Execution
Nested execution refers to the ability to create new work directly from the device, enabling the expression of recursive or data-dependent parallel algorithms in a more natural and easy-to-understand way. It involves launching child grids from within a kernel, which can have multiple levels of nesting. The depth of nested levels and the strategy for launching child grids are important considerations for implementing efficient nested kernels.

- Nested execution allows for the creation of new work directly from the device.
- It enables the expression of recursive or data-dependent parallel algorithms.
- Child grids can be launched from within a kernel, resulting in multiple levels of nesting.
- The depth of nested levels and the strategy for launching child grids are important for efficient implementation.

### 3.6.2 Nested Hello World on the GPU
The concept of dynamic parallelism is introduced in this section, where a kernel is created to print "Hello World" using nested, recursive execution. The parent grid is invoked by the host with 8 threads in a single thread block. Thread 0 in the parent grid then invokes a child grid with half as many threads. This process continues recursively until only one thread is left in the final nesting. The output of the nested kernel program is shown, indicating the recursion depth and the execution from each thread and block.

### 3.6.3 Nested Reduction
Nested reduction is a technique used in parallel computing to perform a reduction operation on a large array of data elements. It involves dividing the input vector into smaller chunks and having each thread calculate the partial sum for its chunk. The partial results from each chunk are then added together to obtain the final sum. This approach reduces or avoids warp divergence, which can negatively impact kernel performance. Different implementations of nested reduction, such as the neighbored pair and interleaved pair approaches, can be used depending on the specific requirements of the reduction operation.

### Chapter 3 Summary
The CUDA execution model on GPU devices has two key features: threads are executed in warps in a Single Instruction Multiple Thread (SIMT) fashion, and hardware resources are partitioned among blocks and threads. These features allow you to control how your application utilizes instruction and memory bandwidth to increase parallelism and performance. Different GPU devices have different hardware limits, so grid and block heuristics are important for optimizing kernel performance. Dynamic parallelism enables the creation of new work directly from the device, allowing for the expression of recursive or data-dependent parallel algorithms. Implementing efficient nested kernels requires attention to the device runtime, including the child grid launch strategy, parent-child synchronization, and the depth of nested levels. In this chapter, you also learned how to analyze kernel performance using the command-line profiling tool, nvprof. Profiling is crucial in CUDA programming to identify performance bottlenecks and optimize kernel behavior.

## Further Chapter 4 Explanation

**Coming Soon!!!** 

## References 

[1] J. Cheng, M. Grossman, and T. McKercher, Professional Cuda C Programming. Indianapolis (Ind.): Wrox, 2014. 
