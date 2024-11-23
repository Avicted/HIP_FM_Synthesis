#include "Includes.hpp"

// 16-bit PCM Output
internal i16
ConvertTo16Bit(f64 sample)
{
    return (i16)((f64)sample * 32767.0f);
}

// 24-bit PCM Output
internal void
Write24BitSample(FILE *file, f64 sample)
{
    i32 intSample = (i32)((f64)sample * 8388607.0f); // Scale to 24-bit
    u8 bytes[3] = {
        (u8)(intSample & 0xFF),
        (u8)((intSample >> 8) & 0xFF),
        (u8)((intSample >> 16) & 0xFF)};
    fwrite(bytes, 1, 3, file);
}

// 32-bit f64 Output
internal void
Write32Bitf64Sample(FILE *file, f64 sample)
{
    f64 floatSample = (f64)sample; // Convert f64 to f32 for 32-bit f32 output
    fwrite(&floatSample, sizeof(f64), 1, file);
}

internal void
WriteWAVHeader(FILE *file, i32 sampleRate, i32 numChannels, i32 bitDepth, i32 numSamples)
{
    i32 byteRate = sampleRate * numChannels * (bitDepth / 8);
    i32 blockAlign = numChannels * (bitDepth / 8);
    i32 dataChunkSize = numSamples * blockAlign;
    i32 fileSize = 36 + dataChunkSize;

    // Write RIFF header
    fwrite("RIFF", 1, 4, file);
    fwrite(&fileSize, 4, 1, file);
    fwrite("WAVE", 1, 4, file);

    // Write fmt subchunk
    fwrite("fmt ", 1, 4, file);
    i32 subchunk1Size = 16; // PCM header size
    fwrite(&subchunk1Size, 4, 1, file);

    short audioFormat = (bitDepth == 32) ? 3 : 1; // 3 = IEEE f64, 1 = PCM
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
            Write32Bitf64Sample(file, sample);
        }
    }

    fclose(file);
    printf("\tWAV file written: %s\n", filename);
}
