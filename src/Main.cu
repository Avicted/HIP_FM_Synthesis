#include <stdio.h>
#include <cmath>
#include <vector>

#include <hip/hip_runtime.h>

#define PI acos(-1.0f)

// Define parameters for the synthesis
const int sampleRate = 44100;
const int signalLength = sampleRate * 5;   // 5 seconds of sound
const float initialCarrierFreq = 440.0f;   // note (440 Hz) for FM synthesis
const float initialModulatorFreq = 220.0f; // Modulation frequency
const float modulationIndex = 1.0f;        // Depth of modulation
const float amplitude = 0.20f;             // Volume

// ADSR (Attack, Decay, Sustain, Release) envelope parameters
const float attackTime = 0.2f;   // Attack duration in seconds
const float decayTime = 0.3f;    // Decay duration in seconds
const float sustainLevel = 0.8f; // Sustain amplitude (0.0 to 1.0)
const float releaseTime = 0.3f;  // Release duration in seconds
const float noteDuration = 3.0f; // Total note duration in seconds

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

__global__ void
HelloWorldKernel(void)
{
    printf("\tHello from CUDA Kernel!\n");
}

__device__ float
ApplyEnvelope(
    float time,
    float attackTime,
    float decayTime,
    float sustainLevel,
    float releaseTime,
    float noteDuration)
{
    if (time < attackTime)
    {
        // Attack phase: Linearly increase amplitude
        return time / attackTime;
    }
    else if (time < attackTime + decayTime)
    {
        // Decay phase: Linearly decrease amplitude to the sustain level
        return 1.0f - ((time - attackTime) / decayTime) * (1.0f - sustainLevel);
    }
    else if (time < noteDuration - releaseTime)
    {
        // Sustain phase: Maintain the sustain level
        return sustainLevel;
    }
    else if (time < noteDuration)
    {
        // Release phase: Linearly decrease amplitude to 0
        return sustainLevel * (1.0f - (time - (noteDuration - releaseTime)) / releaseTime);
    }
    // After release, amplitude is 0
    return 0.0f;
}

// FM Synthesis Kernel
__global__ void
FMSynthesisWithEnvelope(
    float *outputSignal, int sampleRate, int signalLength,
    float carrierFreq, float modulatorFreq, float modulationIndex,
    float amplitude, float attackTime, float decayTime,
    float sustainLevel, float releaseTime, float noteDuration)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < signalLength)
    {
        float time = (float)idx / sampleRate;

        // Vary the frequencies over time (e.g., a slow glide for both carrier and modulator)
        float carrierFreq = initialCarrierFreq + sinf(time * 0.1f) * 50.0f;      // Vary by 50Hz
        float modulatorFreq = initialModulatorFreq + sinf(time * 0.05f) * 25.0f; // Vary by 25Hz

        // Apply the ADSR envelope
        float envelope = ApplyEnvelope(time, attackTime, decayTime, sustainLevel, releaseTime, noteDuration);

        // FM synthesis equation: y(t) = A * sin(2 * pi * f_carrier * t + I * sin(2 * pi * f_modulator * t))
        // Modulation
        float modulator = modulationIndex * sinf(2.0f * M_PI * modulatorFreq * time);

        // Carrier signal with envelope applied
        float signal = envelope * amplitude * sinf(2.0f * M_PI * carrierFreq * time + modulator);

        // Store the result
        outputSignal[idx] = signal;
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

    // Allocate device memory
    float *d_outputSignal;
    HIP_ERRCHK(hipMalloc(&d_outputSignal, signalLength * sizeof(float)));

    // Launch the kernel
    dim3 blockDim(256);
    dim3 gridDim((signalLength + blockDim.x - 1) / blockDim.x);

    FMSynthesisWithEnvelope<<<gridDim, blockDim>>>(
        d_outputSignal, sampleRate, signalLength,
        carrierFreq, modulatorFreq, modulationIndex,
        amplitude,
        attackTime, decayTime, sustainLevel, releaseTime, noteDuration);

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
