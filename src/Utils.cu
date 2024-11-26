#include "Includes.hpp"

// HIP error handling macro
#define HIP_ERRCHK(err) (hip_errchk(err, __FILE__, __LINE__))
internal inline void hip_errchk(hipError_t err, const char *file, i32 line)
{
    if (err != hipSuccess)
    {
        printf("\n\n%s in %s at line %d\n", hipGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

internal i16
GetHIPDevices(void)
{
    i32 deviceCount = 0;
    hipError_t Error = hipGetDeviceCount(&deviceCount);

    if (Error != hipSuccess)
    {
        printf("\tFailed to get HIP Device Count: %s\n", hipGetErrorString(Error));
        return -1;
    }
    else
    {
        printf("\tHIP Device Count: %d\n", deviceCount);
    }

    for (i32 i = 0; i < deviceCount; i++)
    {
        hipDeviceProp_t prop;
        hipError_t Error = hipGetDeviceProperties(&prop, i);

        if (Error != hipSuccess)
        {
            printf("\tFailed to get HIP Device Properties: %s\n", hipGetErrorString(Error));
            continue;
        }
        else
        {
            printf("\tDevice %d: %s\n", i, prop.name);
            printf("\t\tCompute Capability:\t\t\t%d.%d\n", prop.major, prop.minor);
            printf("\t\tTotal Global Memory:\t\t\t%lu\n", prop.totalGlobalMem);
            printf("\t\tShared Memory per Block:\t\t%lu\n", prop.sharedMemPerBlock);
            printf("\t\tRegisters per Block:\t\t\t%d\n", prop.regsPerBlock);
            printf("\t\tWarp Size:\t\t\t\t%d\n", prop.warpSize);
            printf("\t\tMax Threads per Block:\t\t\t%d\n", prop.maxThreadsPerBlock);
            printf("\t\tMax Threads Dimension:\t\t\t(%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
            printf("\t\tMax Grid Size:\t\t\t\t(%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
            printf("\t\tClock Rate:\t\t\t\t%d\n", prop.clockRate);
            printf("\t\tTotal Constant Memory:\t\t\t%lu\n", prop.totalConstMem);
            printf("\t\tMultiprocessor Count:\t\t\t%d\n", prop.multiProcessorCount);
            printf("\t\tL2 Cache Size:\t\t\t\t%d\n", prop.l2CacheSize);
            printf("\t\tMax Threads per Multiprocessor:\t\t%d\n", prop.maxThreadsPerMultiProcessor);
            printf("\t\tUnified Addressing:\t\t\t%d\n", prop.unifiedAddressing);
            printf("\t\tMemory Clock Rate:\t\t\t%d\n", prop.memoryClockRate);
            printf("\t\tMemory Bus Width:\t\t\t%d\n", prop.memoryBusWidth);
            printf("\t\tPeak Memory Bandwidth:\t\t\t%f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        }
    }

    return deviceCount;
}
