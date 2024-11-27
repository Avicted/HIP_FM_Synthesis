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
        printf("    Failed to get HIP Device Count: %s\n", hipGetErrorString(Error));
        return -1;
    }
    else
    {
        printf("    HIP Device Count: %d\n", deviceCount);
    }

    for (i32 i = 0; i < deviceCount; i++)
    {
        hipDeviceProp_t prop;
        hipError_t Error = hipGetDeviceProperties(&prop, i);

        if (Error != hipSuccess)
        {
            printf("    Failed to get HIP Device Properties: %s\n", hipGetErrorString(Error));
            continue;
        }
        else
        {
            printf("    Device %d: %s\n", i, prop.name);
            printf("        Compute Capability: ------------ = %d.%d\n", prop.major, prop.minor);
            printf("        Total Global Memory: ----------- = %lu\n", prop.totalGlobalMem);
            printf("        Shared Memory per Block: ------- = %lu\n", prop.sharedMemPerBlock);
            printf("        Registers per Block: ----------- = %d\n", prop.regsPerBlock);
            printf("        Warp Size: --------------------- = %d\n", prop.warpSize);
            printf("        Max Threads per Block: --------- = %d\n", prop.maxThreadsPerBlock);
            printf("        Max Threads Dimension: --------- = (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
            printf("        Max Grid Size: ----------------- = (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
            printf("        Clock Rate: -------------------- = %d\n", prop.clockRate);
            printf("        Total Constant Memory: --------- = %lu\n", prop.totalConstMem);
            printf("        Multiprocessor Count: ---------- = %d\n", prop.multiProcessorCount);
            printf("        L2 Cache Size: ----------------- = %d\n", prop.l2CacheSize);
            printf("        Max Threads per Multiprocessor:  = %d\n", prop.maxThreadsPerMultiProcessor);
            printf("        Unified Addressing: ------------ = %d\n", prop.unifiedAddressing);
            printf("        Memory Clock Rate: ------------- = %d\n", prop.memoryClockRate);
            printf("        Memory Bus Width: -------------- = %d\n", prop.memoryBusWidth);
            printf("        Peak Memory Bandwidth: --------- = %f\n\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        }
    }

    return deviceCount;
}
