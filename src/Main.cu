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

// -----------------------------------------------------------------------

// Define parameters for the synthesis
const i32 sampleRateHz = 48000; // Default: 48kHz. Allow user input for other rates like 44100, 96000, etc.
i32 signalLengthInSeconds = 0;  // Dynamically set based on MIDI file duration
unsigned long long signalLength = sampleRateHz * signalLengthInSeconds;

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
    i32 sampleRateHz;
    i32 signalLengthInSeconds;
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
    i32 note;      // MIDI note number
    f64 startTime; // Note start time in seconds
    f64 duration;  // Note duration in seconds
    i32 velocity;  // Note velocity
};

// Create host buffer for the output signal
std::vector<f64> outputSignal;

u64 MemoryUsageInBytes = 0;

// -----------------------------------------------------------------------

internal std::vector<MidiNote>
ParseMidi(const std::string &filename, i32 sampleRateHz)
{
    smf::MidiFile midiFile;
    if (!midiFile.read(filename))
    {
        throw std::runtime_error("Failed to load MIDI file: " + filename);
    }

    midiFile.doTimeAnalysis();
    midiFile.linkNotePairs();

    std::vector<MidiNote> notes;
    for (i32 track = 0; track < midiFile.getTrackCount(); ++track)
    {
        for (i32 event = 0; event < midiFile[track].size(); ++event)
        {
            auto &midiEvent = midiFile[track][event];
            if (!midiEvent.isNoteOn())
            {
                continue;
            }

            i32 note = midiEvent.getKeyNumber();
            i32 velocity = midiEvent.getVelocity();
            f64 startTime = midiEvent.seconds;

            if (midiEvent.getLinkedEvent() != nullptr)
            {
                f64 endTime = midiEvent.getLinkedEvent()->seconds;
                notes.push_back({note, static_cast<f64>(startTime),
                                 static_cast<f64>(endTime - startTime), velocity});
            }
        }
    }

    signalLengthInSeconds = midiFile.getFileDurationInSeconds();
    signalLength = sampleRateHz * signalLengthInSeconds;

    return notes;
}

__global__ void
HelloWorldKernel(void)
{
    printf("\tHello from HIP Kernel!\n");
}

__device__ f64
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

__device__ f64
GenerateWaveform(WaveformType type, f64 phase)
{
    if (type == WaveformType::Sine)
    {
        return sin(phase);
    }
    else if (type == WaveformType::Square)
    {
        return fmod(phase, 2.0 * PI) < PI ? 1.0 : -1.0;
    }
    else if (type == WaveformType::Triangle)
    {
        return 2.0 * fabs(2.0 * (phase / (2.0 * PI) - floor(phase / (2.0 * PI) + 0.5))) - 1.0;
    }
    else if (type == WaveformType::Sawtooth)
    {
        return 2.0 * (phase / (2.0 * PI) - floor(phase / (2.0 * PI))) - 1.0;
    }
    else
    {
        return 0.0; // Fallback for undefined types
    }
}

// FM Synthesis Kernel
__global__ void
FMSynthesis(FMSynthParams params, f64 *outputSignal, MidiNote *midiNotes, i32 numNotes)
{
    i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.signalLength)
    {
        return;
    }

    f64 time = (f64)idx / params.sampleRateHz;
    f64 signal = 0.0f;

    for (i32 i = 0; i < numNotes; ++i)
    {
        const MidiNote &note = midiNotes[i];
        if (time < note.startTime || time >= note.startTime + note.duration)
        {
            continue;
        }

        const i32 midiNoteA4 = 69;
        f64 carrierFreq = 440.0f * powf(2.0f, (note.note - midiNoteA4) / 12.0f);
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
RunFMSynthesis(f64 *outputSignal, FMSynthParams params, MidiNote *notes, i32 numNotes)
{
    printf("\tRunning FM Synthesis...\n");

    // Allocate device memory
    f64 *d_outputSignal;
    MidiNote *d_notes;
    HIP_ERRCHK(hipMalloc(&d_outputSignal, params.signalLength * sizeof(f64)));
    HIP_ERRCHK(hipMalloc(&d_notes, numNotes * sizeof(MidiNote)));

    // Copy notes to device memory
    HIP_ERRCHK(hipMemcpy(d_notes, notes, numNotes * sizeof(MidiNote), hipMemcpyHostToDevice));

    // Launch the kernel
    dim3 blockDim(256);
    dim3 gridDim((params.signalLength + blockDim.x - 1) / blockDim.x);

    printf("\n\tLaunching kernel with %d blocks and %d threads per block\n", gridDim.x, blockDim.x);
    printf("\tTotal number of threads: %d\n\n", gridDim.x * blockDim.x);

    FMSynthesis<<<gridDim, blockDim>>>(params, d_outputSignal, d_notes, numNotes);

    HIP_ERRCHK(hipDeviceSynchronize());

    // Copy result back to host
    HIP_ERRCHK(hipMemcpy(outputSignal, d_outputSignal, params.signalLength * sizeof(f64), hipMemcpyDeviceToHost));
    HIP_ERRCHK(hipFree(d_outputSignal));
    HIP_ERRCHK(hipFree(d_notes));

    printf("\tFM Synthesis completed!\n");
}

i32 main(i32 argc, char **argv)
{
    printf("\tHello from HIP!\n");

    // Load MIDI file
    const std::string midiFile = "Sonic the Hedgehog 2 - Chemical Plant Zone.mid";
    std::vector<MidiNote> notes = ParseMidi(midiFile, sampleRateHz);

    // Print the extracted notes
    for (const auto &note : notes)
    {
        // @Note(Victor): Remove this to print all notes
        continue;

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
    MemoryUsageInBytes += outputSignal.size() * sizeof(f64);

    u64 MemoryUsageInMegabytes = MemoryUsageInBytes / Megabytes(1);
    printf("\tMemory Usage: %lu megabytes\n", MemoryUsageInMegabytes);

    hipEvent_t startEvent, stopEvent;
    HIP_ERRCHK(hipEventCreate(&startEvent));
    HIP_ERRCHK(hipEventCreate(&stopEvent));
    HIP_ERRCHK(hipEventRecord(startEvent, 0));

    FMSynthParams params;
    params.sampleRateHz = sampleRateHz;
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
    i32 bitDepth = 32;
    char fileName[50];
    sprintf(fileName, "output_%dbit_%dkHz.wav", bitDepth, params.sampleRateHz / 1000);
    WriteWAVFile(fileName, outputSignal.data(), params.signalLength, params.sampleRateHz, bitDepth);

    // Free host memory
    MemoryUsageInBytes -= outputSignal.size() * sizeof(f64);

    MemoryUsageInMegabytes = MemoryUsageInBytes / Megabytes(1);
    printf("\tMemory Usage: %lu megabytes\n", MemoryUsageInMegabytes);

    outputSignal.clear();

    Assert(MemoryUsageInBytes == 0);

    return 0;
}
