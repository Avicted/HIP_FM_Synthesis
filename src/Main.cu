#include <stdio.h>
#include <cmath>
#include <vector>

#include <hip/hip_runtime.h>

#define PI acos(-1.0f)

// Define parameters for the synthesis
const int sampleRate = 48000;        // Default: 48kHz. Allow user input for other rates like 44100, 96000, etc.
const int signalLengthInSeconds = 5; // 5 seconds of sound
const int signalLength = sampleRate * signalLengthInSeconds;
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

struct FMSynthParams
{
    float carrierFreq;
    float modulatorFreq;
    float modulationIndex;
    float amplitude;

    float attackTime;
    float decayTime;
    float sustainLevel;
    float releaseTime;
    float noteDuration;
};

// Create host buffer for the output signal
std::vector<float> outputSignal(signalLength);

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
    printf("\tHello from HIP Kernel!\n");
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
    float envelope = 0.0f;
    if (time < attackTime)
    {
        envelope = time / attackTime;
    }
    else if (time < attackTime + decayTime)
    {
        envelope = 1.0f - ((time - attackTime) / decayTime) * (1.0f - sustainLevel);
    }
    else if (time < noteDuration - releaseTime)
    {
        envelope = sustainLevel;
    }
    else if (time < noteDuration)
    {
        envelope = sustainLevel * (1.0f - (time - (noteDuration - releaseTime)) / releaseTime);
    }

    return envelope;
}

// FM Synthesis Kernel
__global__ void
FMSynthesisWithEnvelope(FMSynthParams params, float *outputSignal, int sampleRate, int signalLength)
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

    FMSynthParams params;
    params.carrierFreq = carrierFreq;
    params.modulatorFreq = modulatorFreq;
    params.modulationIndex = modulationIndex;
    params.amplitude = amplitude;
    params.attackTime = attackTime;
    params.decayTime = decayTime;
    params.sustainLevel = sustainLevel;
    params.releaseTime = releaseTime;
    params.noteDuration = noteDuration;

    FMSynthesisWithEnvelope<<<gridDim, blockDim>>>(
        params,
        d_outputSignal,
        sampleRate,
        signalLength);

    HIP_ERRCHK(hipDeviceSynchronize());

    // Copy result back to host
    HIP_ERRCHK(hipMemcpy(outputSignal, d_outputSignal, signalLength * sizeof(float), hipMemcpyDeviceToHost));
    HIP_ERRCHK(hipFree(d_outputSignal));

    printf("\tFM Synthesis completed!\n");
}

// 16-bit PCM Output
static int16_t
ConvertTo16Bit(float sample)
{
    return (int16_t)(sample * 32767.0f);
}

// 24-bit PCM Output
static void
Write24BitSample(FILE *file, float sample)
{
    int32_t intSample = (int32_t)(sample * 8388607.0f); // Scale to 24-bit
    uint8_t bytes[3] = {
        (uint8_t)(intSample & 0xFF),
        (uint8_t)((intSample >> 8) & 0xFF),
        (uint8_t)((intSample >> 16) & 0xFF)};
    fwrite(bytes, 1, 3, file);
}

// 32-bit Float Output
static void
Write32BitFloatSample(FILE *file, float sample)
{
    fwrite(&sample, sizeof(float), 1, file);
}

static void
WriteWAVHeader(FILE *file, int sampleRate, int numChannels, int bitDepth, int numSamples)
{
    int byteRate = sampleRate * numChannels * (bitDepth / 8);
    int blockAlign = numChannels * (bitDepth / 8);
    int dataChunkSize = numSamples * blockAlign;
    int fileSize = 36 + dataChunkSize;

    // Write RIFF header
    fwrite("RIFF", 1, 4, file);
    fwrite(&fileSize, 4, 1, file);
    fwrite("WAVE", 1, 4, file);

    // Write fmt subchunk
    fwrite("fmt ", 1, 4, file);
    int subchunk1Size = 16; // PCM header size
    fwrite(&subchunk1Size, 4, 1, file);

    short audioFormat = (bitDepth == 32) ? 3 : 1; // 3 = IEEE float, 1 = PCM
    fwrite(&audioFormat, 2, 1, file);
    fwrite(&numChannels, 2, 1, file);
    fwrite(&sampleRate, 4, 1, file);
    fwrite(&byteRate, 4, 1, file);
    fwrite(&blockAlign, 2, 1, file);
    fwrite(&bitDepth, 2, 1, file);

    // Write data subchunk header
    fwrite("data", 1, 4, file);
    fwrite(&dataChunkSize, 4, 1, file);
}

static void
WriteWAVFile(const char *filename, float *samples, int numSamples, int sampleRate, int bitDepth)
{
    FILE *file = fopen(filename, "wb");
    if (!file)
    {
        printf("Failed to open file for writing\n");
        return;
    }

    int numChannels = 1; // Mono
    WriteWAVHeader(file, sampleRate, numChannels, bitDepth, numSamples);

    for (int i = 0; i < numSamples; i++)
    {
        float sample = samples[i];

        if (bitDepth == 16)
        {
            int16_t pcmSample = ConvertTo16Bit(sample);
            fwrite(&pcmSample, sizeof(int16_t), 1, file);
        }
        else if (bitDepth == 24)
        {
            Write24BitSample(file, sample);
        }
        else if (bitDepth == 32)
        {
            Write32BitFloatSample(file, sample);
        }
    }

    fclose(file);
    printf("WAV file written: %s\n", filename);
}

int main(int argc, char **argv)
{
    printf("\tHello from CUDA!\n");

    GetCudaDevices();

    HIP_ERRCHK(hipSetDevice(0));

    HelloWorldKernel<<<1, 1>>>();

    hipEvent_t startEvent, stopEvent;
    HIP_ERRCHK(hipEventCreate(&startEvent));
    HIP_ERRCHK(hipEventCreate(&stopEvent));
    HIP_ERRCHK(hipEventRecord(startEvent, 0));

    RunFMSynthesis(
        outputSignal.data(),
        signalLength,
        sampleRate,
        initialCarrierFreq,
        initialModulatorFreq,
        modulationIndex,
        amplitude);

    HIP_ERRCHK(hipEventRecord(stopEvent, 0));
    HIP_ERRCHK(hipEventSynchronize(stopEvent));

    float milliseconds = 0.0f;
    HIP_ERRCHK(hipEventElapsedTime(&milliseconds, startEvent, stopEvent));
    printf("Kernel execution time: %.3f ms\n", milliseconds);

    int bitDepth = 16;
    WriteWAVFile("output_32bit_48kHz.wav", outputSignal.data(), signalLength, sampleRate, bitDepth);

    // Free host memory
    outputSignal.clear();

    return 0;
}
