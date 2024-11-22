#include "Includes.hpp"

// 16-bit PCM Output
static i16
ConvertTo16Bit(double sample)
{
    return (i16)((double)sample * 32767.0f);
}

// 24-bit PCM Output
static void
Write24BitSample(FILE *file, double sample)
{
    i32 intSample = (i32)((double)sample * 8388607.0f); // Scale to 24-bit
    u8 bytes[3] = {
        (u8)(intSample & 0xFF),
        (u8)((intSample >> 8) & 0xFF),
        (u8)((intSample >> 16) & 0xFF)};
    fwrite(bytes, 1, 3, file);
}

// 32-bit double Output
static void
Write32BitDoubleSample(FILE *file, double sample)
{
    double floatSample = (double)sample; // Convert double to f32 for 32-bit f32 output
    fwrite(&floatSample, sizeof(double), 1, file);
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

    short audioFormat = (bitDepth == 32) ? 3 : 1; // 3 = IEEE double, 1 = PCM
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
WriteWAVFile(
    const char *filename,
    double *samples,
    unsigned long long numSamples,
    int sampleRate,
    int bitDepth)
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
        double sample = samples[i];

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
            Write32BitDoubleSample(file, sample);
        }
    }

    fclose(file);
    printf("\tWAV file written: %s\n", filename);
}
