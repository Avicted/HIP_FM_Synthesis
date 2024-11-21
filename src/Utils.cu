// HIP error handling macro
#define HIP_ERRCHK(err) (hip_errchk(err, __FILE__, __LINE__))
static inline void hip_errchk(hipError_t err, const char *file, int line)
{
    if (err != hipSuccess)
    {
        printf("\n\n%s in %s at line %d\n", hipGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

static void
GetCudaDevices(void)
{
    int deviceCount = 0;
    hipError_t Error = hipGetDeviceCount(&deviceCount);

    if (Error != hipSuccess)
    {
        printf("\tFailed to get CUDA Device Count: %s\n", hipGetErrorString(Error));
        return;
    }
    else
    {
        printf("\tCUDA Device Count: %d\n", deviceCount);
    }

    for (int i = 0; i < deviceCount; i++)
    {
        hipDeviceProp_t prop;
        hipError_t Error = hipGetDeviceProperties(&prop, i);

        if (Error != hipSuccess)
        {
            printf("\tFailed to get CUDA Device Properties: %s\n", hipGetErrorString(Error));
            continue;
        }
        else
        {
            printf("\tDevice %d: %s\n", i, prop.name);
            printf("\t\tCompute Capability: %d.%d\n", prop.major, prop.minor);
            printf("\t\tTotal Global Memory: %lu\n", prop.totalGlobalMem);
            printf("\t\tShared Memory per Block: %lu\n", prop.sharedMemPerBlock);
            printf("\t\tRegisters per Block: %d\n", prop.regsPerBlock);
            printf("\t\tWarp Size: %d\n", prop.warpSize);
            printf("\t\tMax Threads per Block: %d\n", prop.maxThreadsPerBlock);
            printf("\t\tMax Threads Dimension: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
            printf("\t\tMax Grid Size: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
            printf("\t\tClock Rate: %d\n", prop.clockRate);
            printf("\t\tTotal Constant Memory: %lu\n", prop.totalConstMem);
            printf("\t\tMultiprocessor Count: %d\n", prop.multiProcessorCount);
            printf("\t\tL2 Cache Size: %d\n", prop.l2CacheSize);
            printf("\t\tMax Threads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
            printf("\t\tUnified Addressing: %d\n", prop.unifiedAddressing);
            printf("\t\tMemory Clock Rate: %d\n", prop.memoryClockRate);
            printf("\t\tMemory Bus Width: %d\n", prop.memoryBusWidth);
            printf("\t\tPeak Memory Bandwidth: %f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        }
    }
}
