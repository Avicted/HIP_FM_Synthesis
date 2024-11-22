// 16-bit PCM Output
static int16_t
ConvertTo16Bit(double sample)
{
    return (int16_t)((double)sample * 32767.0f);
}

// 24-bit PCM Output
static void
Write24BitSample(FILE *file, double sample)
{
    int32_t intSample = (int32_t)((double)sample * 8388607.0f); // Scale to 24-bit
    uint8_t bytes[3] = {
        (uint8_t)(intSample & 0xFF),
        (uint8_t)((intSample >> 8) & 0xFF),
        (uint8_t)((intSample >> 16) & 0xFF)};
    fwrite(bytes, 1, 3, file);
}

// 32-bit double Output
static void
Write32BitDoubleSample(FILE *file, double sample)
{
    double floatSample = (double)sample; // Convert double to float for 32-bit float output
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
            int16_t pcmSample = ConvertTo16Bit(sample);
            fwrite(&pcmSample, sizeof(int16_t), 1, file);
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
