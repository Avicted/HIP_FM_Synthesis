#include <stdio.h>
#include <cmath>
#include <vector>

#include <hip/hip_runtime.h>

#define PI acos(-1.0f)

// Define parameters for the synthesis
const int sampleRate = 44100;
const int signalLength = sampleRate * 5;   // 5 seconds of sound
const float initialCarrierFreq = 150.0f;   // note (150 Hz) for FM synthesis
const float initialModulatorFreq = 120.0f; // Modulation frequency
const float modulationIndex = 1.0f;        // Depth of modulation
const float amplitude = 0.25f;             // Volume

// Create host buffer for the output signal
std::vector<float> FMSignal(signalLength);

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

__global__ void HelloWorldKernel(void)
{
    printf("\tHello from CUDA Kernel!\n");
}

// FM Synthesis Kernel
__global__ void FMSynthesis(
    float *outputSignal,
    int sampleRate,
    int signalLength,
    float carrierFreq,
    float modulatorFreq,
    float modulationIndex,
    float amplitude)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < signalLength)
    {
        // Time in seconds for the current sample
        float time = (float)idx / (float)sampleRate;

        // Vary the frequencies over time (e.g., a slow glide for both carrier and modulator)
        float carrierFreq = initialCarrierFreq + sinf(time * 0.1f) * 50.0f;      // Vary by 50Hz
        float modulatorFreq = initialModulatorFreq + sinf(time * 0.05f) * 25.0f; // Vary by 25Hz

        // FM synthesis equation: y(t) = A * sin(2 * pi * f_carrier * t + I * sin(2 * pi * f_modulator * t))
        float modulator = modulationIndex * sinf(2 * PI * modulatorFreq * time);
        outputSignal[idx] = amplitude * sinf(2 * PI * carrierFreq * time + modulator);
    }
}

static void
RunFMSynthesis(
    float *outputSignal,
    int signalLength,
    int sampleRate,
    float carrierFreq,
    float modulatorFreq,
    float modulationIndex,
    float amplitude)
{
    printf("\tRunning FM Synthesis...\n");

    float *d_outputSignal;
    HIP_ERRCHK(hipMalloc((void **)&d_outputSignal, signalLength * sizeof(float)));

    int threadsPerBlock = 256;
    int blocksPerGrid = (signalLength + threadsPerBlock - 1) / threadsPerBlock;

    // Launch FM synthesis kernel
    FMSynthesis<<<blocksPerGrid, threadsPerBlock>>>(d_outputSignal, sampleRate, signalLength, carrierFreq, modulatorFreq, modulationIndex, amplitude);

    HIP_ERRCHK(hipDeviceSynchronize());

    // Copy result back to host
    HIP_ERRCHK(hipMemcpy(outputSignal, d_outputSignal, signalLength * sizeof(float), hipMemcpyDeviceToHost));
    HIP_ERRCHK(hipFree(d_outputSignal));

    printf("\tFM Synthesis completed!\n");
}

static void
SaveSignalToWAV(
    const char *filename,
    float *signal,
    int sampleRate,
    int signalLength)
{
    printf("\tSaving signal to WAV file...\n");

    // Save the output signal to a .wav file
    FILE *file = fopen("FM_Synthesis.wav", "wb");
    if (file)
    {
        // Write the header
        int bitsPerSample = 16;
        int byteRate = sampleRate * bitsPerSample / 8;
        int blockAlign = bitsPerSample / 8;
        int dataSize = signalLength * blockAlign;

        fwrite("RIFF", 1, 4, file);
        int fileSize = 36 + dataSize;
        fwrite(&fileSize, 4, 1, file);
        fwrite("WAVE", 1, 4, file);
        fwrite("fmt ", 1, 4, file);
        int fmtSize = 16;
        fwrite(&fmtSize, 4, 1, file);
        short format = 1;
        fwrite(&format, 2, 1, file);
        short channels = 1;
        fwrite(&channels, 2, 1, file);
        fwrite(&sampleRate, 4, 1, file);
        fwrite(&byteRate, 4, 1, file);
        fwrite(&blockAlign, 2, 1, file);
        fwrite(&bitsPerSample, 2, 1, file);
        fwrite("data", 1, 4, file);
        fwrite(&dataSize, 4, 1, file);

        // Write the audio data
        for (int i = 0; i < signalLength; i++)
        {
            short sample = (short)(FMSignal[i] * 32767.0f);
            fwrite(&sample, 2, 1, file);
        }

        fclose(file);
    }
}

int main(int argc, char **argv)
{
    printf("\tHello from CUDA!\n");

    GetCudaDevices();

    HIP_ERRCHK(hipSetDevice(0));

    HelloWorldKernel<<<1, 1>>>();

    // Start to measure time
    auto start = std::chrono::high_resolution_clock::now();

    RunFMSynthesis(
        FMSignal.data(),
        signalLength,
        sampleRate,
        initialCarrierFreq,
        initialModulatorFreq,
        modulationIndex,
        amplitude);

    SaveSignalToWAV("FM_Synthesis.wav", FMSignal.data(), sampleRate, signalLength);

    // Stop measuring time
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("\tElapsed time: %f seconds\n", elapsed.count());
    printf("\tElapsed time: %f milliseconds\n", elapsed.count() * 1000);

    // Free host memory
    FMSignal.clear();

    return 0;
}
