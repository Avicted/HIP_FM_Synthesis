#include "Includes.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdint>

#define WAVE_FORMAT_PCM 1
#define WAVE_FORMAT_IEEE_FLOAT 3

// 16-bit PCM Output
internal i16
ConvertTo16Bit(f64 sample)
{
    const f64 SCALE_16BIT = 32767.0f;        // Half of the 16-bit range
    return (i16)((f64)sample * SCALE_16BIT); // Scale to 16-bit
}

// 24-bit PCM Output
internal void
Write24BitSample(FILE *file, f64 sample)
{
    const f64 SCALE_24BIT = 8388607.0f;          // Half of the 24-bit range
    i32 intSample = (i32)(sample * SCALE_24BIT); // Scale to 24-bit
    u8 bytes[3] = {
        (u8)(intSample & 0xFF),
        (u8)((intSample >> 8) & 0xFF),
        (u8)((intSample >> 16) & 0xFF)};
    fwrite(bytes, 1, 3, file);
}

internal f64
NormalizeSample(f64 sample)
{
    // Ensure the sample stays within the -1.0 to 1.0 range
    return std::min(std::max(sample, -1.0), 1.0);
}

// 32-bit PCM Output
internal void
Write32BitSample(FILE *file, f64 sample)
{
    f32 floatSample = (f32)NormalizeSample(sample);
    fwrite(&floatSample, sizeof(f32), 1, file);
}

internal void
WriteWAVHeader(FILE *file, i32 sampleRate, i32 numChannels, i32 bitDepth, i32 numSamples)
{
    i32 byteRate = sampleRate * numChannels * (bitDepth / 8); // Sample rate * channels * bytes per sample
    i32 blockAlign = numChannels * (bitDepth / 8);            // Channels * bytes per sample
    i32 dataChunkSize = numSamples * blockAlign;              // Num samples * bytes per sample
    i32 fileSize = 36 + dataChunkSize;                        // File size = header size (36) + data chunk size

    // Write RIFF header
    fwrite("RIFF", 1, 4, file);    // 'RIFF' chunk descriptor
    fwrite(&fileSize, 4, 1, file); // File size - 8 bytes (total file size minus the 'RIFF' and size fields)
    fwrite("WAVE", 1, 4, file);    // 'WAVE' format

    // Write fmt subchunk header
    i32 subchunk1Size = 16;             // PCM header size is always 16 for basic PCM/float formats
    fwrite("fmt ", 1, 4, file);         // 'fmt ' subchunk identifier
    fwrite(&subchunk1Size, 4, 1, file); // Subchunk1 size (16 for PCM/float formats)

    short audioFormat = (bitDepth == 32) ? WAVE_FORMAT_IEEE_FLOAT : WAVE_FORMAT_PCM; // Audio format type: 1 = PCM, 3 = IEEE float
    fwrite(&audioFormat, 2, 1, file);                                                // Audio format: 1 (PCM) or 3 (IEEE float)
    fwrite(&numChannels, 2, 1, file);                                                // Number of channels
    fwrite(&sampleRate, 4, 1, file);                                                 // Sample rate (e.g., 44100, 48000)
    fwrite(&byteRate, 4, 1, file);                                                   // Byte rate (sampleRate * numChannels * bytesPerSample)
    fwrite(&blockAlign, 2, 1, file);                                                 // Block align (numChannels * bytesPerSample)
    fwrite(&bitDepth, 2, 1, file);                                                   // Bits per sample (e.g., 16, 24, 32)

    // Write data subchunk header
    fwrite("data", 1, 4, file);         // 'data' subchunk identifier
    fwrite(&dataChunkSize, 4, 1, file); // Data chunk size (number of samples * bytes per sample)
}

internal void
WriteWAVFile(
    const char *filename,
    f64 *samples,
    unsigned long long numSamples,
    i32 sampleRate,
    i32 bitDepth)
{
    FILE *file = fopen(filename, "wb");
    if (!file)
    {
        printf("Failed to open file for writing\n");
        return;
    }

    i32 numChannels = 1; // Mono
    WriteWAVHeader(file, sampleRate, numChannels, bitDepth, numSamples);

    for (i32 i = 0; i < numSamples; i++)
    {
        f64 sample = samples[i];

        if (bitDepth == 16)
        {
            i16 pcmSample = ConvertTo16Bit(sample);
            fwrite(&pcmSample, sizeof(i16), 1, file);
        }
        else if (bitDepth == 24)
        {
            Write24BitSample(file, sample);
        }
        else if (bitDepth == 32)
        {
            Write32BitSample(file, sample);
        }
    }

    fclose(file);
    printf("\tWAV file written: %s\n", filename);
}
