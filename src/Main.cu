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
#include "Includes.hpp"
#include "Utils.cu"
#include "WAV_Helper.cu"

// Define parameters for the synthesis
const int sampleRate = 48000;   // Default: 48kHz. Allow user input for other rates like 44100, 96000, etc.
int signalLengthInSeconds = 20; // 20 seconds of sound
unsigned long long signalLength = sampleRate * signalLengthInSeconds;

const f64 initialCarrierFreq = 440.0f;   // note (440 Hz) for FM synthesis
const f64 initialModulatorFreq = 880.0f; // Modulation frequency
const f64 modulationIndex = 0.1f;        // Depth of modulation
const f64 amplitude = 0.20f;             // Volume

// ADSR (Attack, Decay, Sustain, Release) envelope parameters
const f64 attackTime = 0.0050f; // Attack duration in seconds
const f64 decayTime = 0.40f;    // Decay duration in seconds
const f64 sustainLevel = 0.50f; // Sustain amplitude (0.0 to 1.0)
const f64 releaseTime = 0.20f;  // Release duration in seconds
const f64 noteDuration = 0.0f;  // = signalLengthInSeconds;

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

    f64 carrierFreq;
    f64 modulatorFreq;
    f64 modulationIndex;
    f64 amplitude;

    f64 attackTime;
    f64 decayTime;
    f64 sustainLevel;
    f64 releaseTime;
    f64 noteDuration;

    WaveformType waveformType;
};

struct MidiNote
{
    int note;      // MIDI note number
    f64 startTime; // Note start time in seconds
    f64 duration;  // Note duration in seconds
    int velocity;  // Note velocity
};

// Create host buffer for the output signal
std::vector<double> outputSignal;

// -----------------------------------------------------------------------

internal std::vector<MidiNote>
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
            f64 startTime = midiEvent.seconds;

            if (midiEvent.getLinkedEvent() != nullptr)
            {
                f64 endTime = midiEvent.getLinkedEvent()->seconds;
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
    f64 time,
    f64 attackTime,
    f64 decayTime,
    f64 sustainLevel,
    f64 releaseTime,
    f64 noteDuration)
{
    f64 envelope = 0.0f;
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
GenerateWaveform(WaveformType type, f64 phase)
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
FMSynthesis(FMSynthParams params, f64 *outputSignal, MidiNote *midiNotes, int numNotes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.signalLength)
    {
        return;
    }

    f64 time = (double)idx / params.sampleRate;
    f64 signal = 0.0f;

    for (int i = 0; i < numNotes; ++i)
    {
        const MidiNote &note = midiNotes[i];
        if (time < note.startTime || time >= note.startTime + note.duration)
        {
            continue;
        }

        f64 carrierFreq = 440.0f * powf(2.0f, (note.note - 69) / 12.0f);
        f64 modulatorFreq = carrierFreq * 2.0; // params.modulatorFreq; // Can vary per note if needed
        f64 phase = fmodf(2.0f * PI * carrierFreq * time, 2.0f * PI);
        phase += params.modulationIndex * __sinf(2.0f * PI * modulatorFreq * time);

        f64 envelope = ApplyEnvelope(
            time - note.startTime,
            params.attackTime,
            params.decayTime,
            params.sustainLevel,
            params.releaseTime,
            note.duration);

        // Signal accumulation
        signal += note.velocity / 127.0 * params.amplitude * envelope *
                  GenerateWaveform(params.waveformType, phase);

        f64 waveform = GenerateWaveform(params.waveformType, phase);

        // Apply envelope
        signal *= envelope;

        outputSignal[idx] = signal;
    }
}

internal void
RunFMSynthesis(f64 *outputSignal, FMSynthParams params, MidiNote *notes, int numNotes)
{
    printf("\tRunning FM Synthesis...\n");

    // Allocate device memory
    f64 *d_outputSignal;
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

    f32 milliseconds = 0.0f;
    HIP_ERRCHK(hipEventElapsedTime(&milliseconds, startEvent, stopEvent));
    printf("\tKernel execution time: %.3f ms\n", milliseconds);

    // 16, 24, 32-bit WAV output
    int bitDepth = 24;
    char fileName[50];
    sprintf(fileName, "output_%dbit_%dkHz.wav", bitDepth, params.sampleRate / 1000);
    WriteWAVFile(fileName, outputSignal.data(), params.signalLength, params.sampleRate, bitDepth);

    // Free host memory
    outputSignal.clear();

    return 0;
}
