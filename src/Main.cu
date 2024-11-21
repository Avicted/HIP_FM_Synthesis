// Standard library headers
#include <stdio.h>
#include <math.h>
#include <vector>

// HIP header
#include <hip/hip_runtime.h>

// Our code
#include "Utils.cu"
#include "WAV_Helper.cu"

#define PI acos(-1.0f)

// Define parameters for the synthesis
const int sampleRate = 48000;         // Default: 48kHz. Allow user input for other rates like 44100, 96000, etc.
const int signalLengthInSeconds = 10; // 10 seconds of sound
const unsigned long long signalLength = sampleRate * signalLengthInSeconds;

const float initialCarrierFreq = 220.0f;   // note (220 Hz) for FM synthesis
const float initialModulatorFreq = 110.0f; // Modulation frequency
const float modulationIndex = 1.0f;        // Depth of modulation
const float amplitude = 0.20f;             // Volume

// ADSR (Attack, Decay, Sustain, Release) envelope parameters
const float attackTime = 0.3f;   // Attack duration in seconds
const float decayTime = 0.3f;    // Decay duration in seconds
const float sustainLevel = 0.7f; // Sustain amplitude (0.0 to 1.0)
const float releaseTime = 1.0f;  // Release duration in seconds
const float noteDuration = signalLengthInSeconds;

enum class WaveformType
{
    Sine,
    Square,
    Triangle,
    Sawtooth
};

struct FMSynthParams
{
    int sampleRate;
    int signalLengthInSeconds;
    unsigned long long signalLength;

    float carrierFreq;
    float modulatorFreq;
    float modulationIndex;
    float amplitude;

    float attackTime;
    float decayTime;
    float sustainLevel;
    float releaseTime;
    float noteDuration;

    WaveformType waveformType;
};

// Create host buffer for the output signal
std::vector<float> outputSignal;

// -----------------------------------------------------------------------

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

__device__ float
GenerateWaveform(WaveformType type, float phase)
{
    switch (type)
    {
    case WaveformType::Sine:
        return sinf(phase);
    case WaveformType::Square:
        return fmodf(phase, 2.0f * PI) < PI ? 1.0f : -1.0f;
    case WaveformType::Triangle:
        return 2.0f * fabsf(2.0f * (phase / (2.0f * PI) - floorf(phase / (2.0f * PI) + 0.5f))) - 1.0f;
    case WaveformType::Sawtooth:
        return 2.0f * (phase / (2.0f * PI) - floorf(phase / (2.0f * PI))) - 1.0f;
    default:
        return 0.0f; // Fallback for undefined types
    }
}

// FM Synthesis Kernel
__global__ void
FMSynthesisWithEnvelope(FMSynthParams params, float *outputSignal)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < params.signalLength)
    {
        float time = (float)idx / params.sampleRate;
        float phase = 2.0f * PI * params.carrierFreq * time;

        // Vary the frequencies over time (e.g., a slow glide for both carrier and modulator)
        float carrierFreq = initialCarrierFreq + __sinf(time * 0.1f) * 1200.0f;     // Vary by 1.2kHz
        float modulatorFreq = initialModulatorFreq + __sinf(time * 0.05f) * 600.0f; // Vary by 600Hz

        // Modulate the carrier frequency
        phase += params.modulationIndex * __sinf(2.0f * PI * modulatorFreq * time);

        // Apply the envelope
        float envelope = ApplyEnvelope(
            time,
            params.attackTime,
            params.decayTime,
            params.sustainLevel,
            params.releaseTime,
            params.noteDuration);

        // Generate the waveform
        float signal = params.amplitude * envelope * GenerateWaveform(params.waveformType, phase);

        // Store the result
        outputSignal[idx] = signal;
    }
}

static void
RunFMSynthesis(float *outputSignal, FMSynthParams params)
{
    printf("\tRunning FM Synthesis...\n");

    // Allocate device memory
    float *d_outputSignal;
    HIP_ERRCHK(hipMalloc(&d_outputSignal, params.signalLength * sizeof(float)));

    // Launch the kernel
    dim3 blockDim(256);
    dim3 gridDim((params.signalLength + blockDim.x - 1) / blockDim.x);

    FMSynthesisWithEnvelope<<<gridDim, blockDim>>>(params, d_outputSignal);

    HIP_ERRCHK(hipDeviceSynchronize());

    // Copy result back to host
    HIP_ERRCHK(hipMemcpy(outputSignal, d_outputSignal, params.signalLength * sizeof(float), hipMemcpyDeviceToHost));
    HIP_ERRCHK(hipFree(d_outputSignal));

    printf("\tFM Synthesis completed!\n");
}

int main(int argc, char **argv)
{
    printf("\tHello from HIP!\n");

    GetCudaDevices();

    HIP_ERRCHK(hipSetDevice(0));

    HelloWorldKernel<<<1, 1>>>();

    // Allocate host memory
    outputSignal.resize(signalLength);

    hipEvent_t startEvent, stopEvent;
    HIP_ERRCHK(hipEventCreate(&startEvent));
    HIP_ERRCHK(hipEventCreate(&stopEvent));
    HIP_ERRCHK(hipEventRecord(startEvent, 0));

    FMSynthParams params;
    params.sampleRate = sampleRate;
    params.signalLengthInSeconds = signalLengthInSeconds;
    params.signalLength = signalLength;

    params.carrierFreq = initialCarrierFreq;
    params.modulatorFreq = initialModulatorFreq;
    params.modulationIndex = modulationIndex;
    params.amplitude = amplitude;

    params.attackTime = attackTime;
    params.decayTime = decayTime;
    params.sustainLevel = sustainLevel;
    params.releaseTime = releaseTime;
    params.noteDuration = noteDuration;
    params.waveformType = WaveformType::Triangle;

    // Setup the memory and call the kernel
    RunFMSynthesis(outputSignal.data(), params);

    HIP_ERRCHK(hipEventRecord(stopEvent, 0));
    HIP_ERRCHK(hipEventSynchronize(stopEvent));

    float milliseconds = 0.0f;
    HIP_ERRCHK(hipEventElapsedTime(&milliseconds, startEvent, stopEvent));
    printf("\tKernel execution time: %.3f ms\n", milliseconds);

    int bitDepth = 32;
    char fileName[50];
    sprintf(fileName, "output_%dbit_%dkHz.wav", bitDepth, params.sampleRate / 1000);
    WriteWAVFile(fileName, outputSignal.data(), params.signalLength, params.sampleRate, bitDepth);

    // Free host memory
    outputSignal.clear();

    return 0;
}
