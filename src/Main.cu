// Standard library headers
#include <stdio.h>
#include <cmath>
#include <vector>
#include <iostream>

// HIP header
#include <hip/hip_runtime.h>

// Midifile submodule
#include "MidiFile.h"

// Our code
#include "Utils.cu"
#include "WAV_Helper.cu"

#define PI acos(-1.0f)

// Define parameters for the synthesis
const int sampleRate = 48000;   // Default: 48kHz. Allow user input for other rates like 44100, 96000, etc.
int signalLengthInSeconds = 20; // 20 seconds of sound
unsigned long long signalLength = sampleRate * signalLengthInSeconds;

const double initialCarrierFreq = 440.0f;   // note (440 Hz) for FM synthesis
const double initialModulatorFreq = 880.0f; // Modulation frequency
const double modulationIndex = 0.1f;        // Depth of modulation
const double amplitude = 0.20f;             // Volume

// ADSR (Attack, Decay, Sustain, Release) envelope parameters
const double attackTime = 0.0050f; // Attack duration in seconds
const double decayTime = 0.40f;    // Decay duration in seconds
const double sustainLevel = 0.50f; // Sustain amplitude (0.0 to 1.0)
const double releaseTime = 0.20f;  // Release duration in seconds
const double noteDuration = 0.0f;  // = signalLengthInSeconds;

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

    double carrierFreq;
    double modulatorFreq;
    double modulationIndex;
    double amplitude;

    double attackTime;
    double decayTime;
    double sustainLevel;
    double releaseTime;
    double noteDuration;

    WaveformType waveformType;
};

struct MidiNote
{
    int note;         // MIDI note number
    double startTime; // Note start time in seconds
    double duration;  // Note duration in seconds
    int velocity;     // Note velocity
};

// Create host buffer for the output signal
std::vector<double> outputSignal;

// -----------------------------------------------------------------------

// Function to load and parse MIDI notes
static std::vector<MidiNote>
ParseMidi(const std::string &filename, int sampleRate)
{
    smf::MidiFile midiFile;
    if (!midiFile.read(filename))
    {
        throw std::runtime_error("Failed to load MIDI file: " + filename);
    }

    midiFile.doTimeAnalysis();
    midiFile.linkNotePairs();

    std::vector<MidiNote> notes;
    for (int track = 0; track < midiFile.getTrackCount(); ++track)
    {
        for (int event = 0; event < midiFile[track].size(); ++event)
        {
            auto &midiEvent = midiFile[track][event];
            if (!midiEvent.isNoteOn())
            {
                continue;
            }

            int note = midiEvent.getKeyNumber();
            int velocity = midiEvent.getVelocity();
            double startTime = midiEvent.seconds;

            if (midiEvent.getLinkedEvent() != nullptr)
            {
                double endTime = midiEvent.getLinkedEvent()->seconds;
                notes.push_back({note, static_cast<double>(startTime),
                                 static_cast<double>(endTime - startTime), velocity});
            }
        }
    }

    // Set project parameters
    signalLengthInSeconds = midiFile.getFileDurationInSeconds();
    signalLength = sampleRate * signalLengthInSeconds;

    return notes;
}

__global__ void
HelloWorldKernel(void)
{
    printf("\tHello from HIP Kernel!\n");
}

__device__ double
ApplyEnvelope(
    double time,
    double attackTime,
    double decayTime,
    double sustainLevel,
    double releaseTime,
    double noteDuration)
{
    double envelope = 0.0f;
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
    else
    {
        envelope = 0.0f;
    }

    return envelope;
}

__device__ double
GenerateWaveform(WaveformType type, double phase)
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
FMSynthesis(FMSynthParams params, double *outputSignal, MidiNote *midiNotes, int numNotes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.signalLength)
    {
        return;
    }

    double time = (double)idx / params.sampleRate;
    double signal = 0.0f;

    for (int i = 0; i < numNotes; ++i)
    {
        const MidiNote &note = midiNotes[i];
        if (time < note.startTime || time >= note.startTime + note.duration)
        {
            continue;
        }

        double carrierFreq = 440.0f * powf(2.0f, (note.note - 69) / 12.0f);
        double modulatorFreq = carrierFreq * 2.0; // params.modulatorFreq; // Can vary per note if needed
        double phase = fmodf(2.0f * PI * carrierFreq * time, 2.0f * PI);
        phase += params.modulationIndex * __sinf(2.0f * PI * modulatorFreq * time);

        double envelope = ApplyEnvelope(
            time - note.startTime,
            params.attackTime,
            params.decayTime,
            params.sustainLevel,
            params.releaseTime,
            note.duration);

        // Signal accumulation
        signal += note.velocity / 127.0 * params.amplitude * envelope *
                  GenerateWaveform(params.waveformType, phase);

        double waveform = GenerateWaveform(params.waveformType, phase);

        // Apply envelope
        signal *= envelope;

        // Output signal
        outputSignal[idx] = signal;
    }
}

static void
RunFMSynthesis(double *outputSignal, FMSynthParams params, MidiNote *notes, int numNotes)
{
    printf("\tRunning FM Synthesis...\n");

    // Allocate device memory
    double *d_outputSignal;
    MidiNote *d_notes;
    HIP_ERRCHK(hipMalloc(&d_outputSignal, params.signalLength * sizeof(double)));
    HIP_ERRCHK(hipMalloc(&d_notes, numNotes * sizeof(MidiNote)));

    // Copy notes to device memory
    HIP_ERRCHK(hipMemcpy(d_notes, notes, numNotes * sizeof(MidiNote), hipMemcpyHostToDevice));

    // Launch the kernel
    dim3 blockDim(256);
    dim3 gridDim((params.signalLength + blockDim.x - 1) / blockDim.x);

    FMSynthesis<<<gridDim, blockDim>>>(params, d_outputSignal, d_notes, numNotes);

    HIP_ERRCHK(hipDeviceSynchronize());

    // Copy result back to host
    HIP_ERRCHK(hipMemcpy(outputSignal, d_outputSignal, params.signalLength * sizeof(double), hipMemcpyDeviceToHost));
    HIP_ERRCHK(hipFree(d_outputSignal));
    HIP_ERRCHK(hipFree(d_notes));

    printf("\tFM Synthesis completed!\n");
}

int main(int argc, char **argv)
{
    printf("\tHello from HIP!\n");

    // Load MIDI file
    const std::string midiFile = "Sonic the Hedgehog 2 - Chemical Plant Zone.mid";
    std::vector<MidiNote> notes = ParseMidi(midiFile, sampleRate);

    // Print the extracted notes
    for (const auto &note : notes)
    {
        std::cout << "Note: " << note.note
                  << ", Start Time: " << note.startTime
                  << ", Duration: " << note.duration
                  << ", Velocity: " << note.velocity << std::endl;
    }

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
    RunFMSynthesis(outputSignal.data(), params, notes.data(), notes.size());

    HIP_ERRCHK(hipEventRecord(stopEvent, 0));
    HIP_ERRCHK(hipEventSynchronize(stopEvent));

    float milliseconds = 0.0f;
    HIP_ERRCHK(hipEventElapsedTime(&milliseconds, startEvent, stopEvent));
    printf("\tKernel execution time: %.3f ms\n", milliseconds);

    // 16, 24, 32-bit WAV output
    int bitDepth = 16;
    char fileName[50];
    sprintf(fileName, "output_%dbit_%dkHz.wav", bitDepth, params.sampleRate / 1000);
    WriteWAVFile(fileName, outputSignal.data(), params.signalLength, params.sampleRate, bitDepth);

    // Free host memory
    outputSignal.clear();

    return 0;
}
