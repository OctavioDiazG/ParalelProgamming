# Investigation CUDA Cores, Threads, Blocks and Grids

## Brief concept introduction 

**CUDA** Is a parallel computing platform and programming model developed by NVIDIA. It allows developers to use a CUDA-enabled graphics processing unit (GPU) for general-purpose processing.

**CUDA Core** is the basic processing unit of the GPU. It is responsible for executing instructions in parallel. A GPU can have thousands of CUDA cores. Interesting fact: The number of CUDA cores in a GPU can vary widely depending on the model. For example, the NVIDIA GeForce GTX 1080 Ti has 3584 CUDA cores, while the NVIDIA Tesla V100 has 5120 CUDA cores.
 
**Thread** is the smallest unit of parallelism in a CUDA program. A thread is executed by a CUDA core. Threads are organized into blocks. Threads are lightweight and can be created and destroyed quickly, which makes them ideal for parallel processing tasks that require frequent creation and destruction of threads. This is particularly useful in scientific simulations where the number of threads needed can vary widely depending on the complexity of the simulation.
 
**Block** is the second level of parallelism in a CUDA program. It is a group of threads that can be executed independently. Blocks are organized into grids. The number of threads in a block is limited by the hardware resources available on the GPU, such as shared memory and registers. This means that careful optimization of block size is necessary to achieve maximum performance in scientific simulations and other parallel processing tasks.

**Grid** is the highest level of parallelism in a CUDA program. It is a group of blocks that can be executed independent. The maximum number of blocks in a grid is limited by the hardware resources available on the GPU, such as the number of multiprocessors and the amount of memory available on the device. This means that careful optimization of grid size is necessary to achieve maximum performance in scientific simulations and other parallel processing tasks.

![image](https://github.com/OctavioDiazG/Parallel_Progamming_ODG/assets/113312422/b23bc656-f9f4-4bcb-86e3-07c81806db8c)

##  Main differences between these concepts 

**CUDA Core vs Thread:** The basic difference between these two concepts is that a CUDA core is a physical processing unit, while a thread is a logical unit of execution. A CUDA core executes instructions in parallel, while a thread performs a single task.

**Thread vs Block:** Threads are organized into blocks. A block is a group of threads that can be executed independently. The main difference between these two concepts is that threads are the smallest unit of parallelism, while blocks are the second level of parallelism.

**Block vs Grid:** Blocks are organized into grids. A grid is a group of blocks that can be executed independently. The main difference between these two concepts is that blocks are the second level of parallelism, while grids are the highest level of parallelism.



## References 

[1] BizhanBizhan 16.2k99 gold badges6363 silver badges101101 bronze badges, “Understanding cuda grid dimensions, block dimensions and threads organization (simple explanation),” Stack Overflow, https://stackoverflow.com/questions/2392250/understanding-cuda-grid-dimensions-block-dimensions-and-threads-organization-s (accessed Oct. 2, 2023). 

[2] cazint, “Optimal threads vs blocks,” NVIDIA Developer Forums, https://forums.developer.nvidia.com/t/optimal-threads-vs-blocks/22124 (accessed Oct. 3, 2023). 

[3] N. Wilt, The Cuda Handbook: A Comprehensive Guide to GPU Programming. Boston: Addison-Wesley, 2019. 

[4] TheBeard and TheBeard, “Cuda - threads, blocks, grids and synchronization,” The Beard Sage, http://thebeardsage.com/cuda-threads-blocks-grids-and-synchronization/ (accessed Oct. 3, 2023).

[5] J. Sanders, E. Kandrot, and J. J. Dongarra, Cuda by Example: An Introduction to General-Purpose GPU Programming. Upper Saddle River etc.: Addison-Wesley/Pearson Education, 2015. 
