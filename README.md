# FM Synthesizer with HIP (ROCm and CUDA)

This project demonstrates **FM Synthesis** (Frequency Modulation) using **HIP** (Heterogeneous Compute), enabling high-performance sound generation on both AMD and NVIDIA GPUs.

## Key Features
FM Synthesis:

 - Implements basic **FM synthesis** to generate sounds by modulating one oscillator's frequency (carrier) using another oscillator (modulator).

 - **Dynamic Frequency Modulation**: Both carrier and modulator frequencies vary over time.

 - Audio data is generated and then written to a .wav file for playback or further processing.

 - Supports 16, 24 and 32 -bit PCM Output at variable sample rates.

 - Multiple waveforms supported: Sine, Square, Triangle, Sawtooth.

 - Can read, parse and generate audio data from a MIDI file.

## Dependencies
 - Ensure that your system supports HIP (either via AMD ROCm or via NVIDIA CUDA).
 - Make

## Build and run
```bash
git clone --recurse-submodules git@github.com:Avicted/HIP_FM_Synthesis.git

make all
make run
```

## Program Output
The following command generates a .wav file with a 1-second sound clip using the default parameters:
```bash
./build/Main
        Hello from HIP!
        CUDA Device Count: 1
        Device 0: AMD Radeon RX 6900 XT
                Compute Capability: 10.3
                Total Global Memory: 17163091968
                Shared Memory per Block: 65536
                Registers per Block: 65536
                Warp Size: 32
                Max Threads per Block: 1024
                Max Threads Dimension: (1024, 1024, 1024)
                Max Grid Size: (2147483647, 65536, 65536)
                Clock Rate: 2660000
                Total Constant Memory: 2147483647
                Multiprocessor Count: 40
                L2 Cache Size: 4194304
                Max Threads per Multiprocessor: 2048
                Unified Addressing: 0
                Memory Clock Rate: 1000000
                Memory Bus Width: 256
                Peak Memory Bandwidth: 64.000000
        Hello from HIP Kernel!
        Memory Usage: 40 megabytes
        Running FM Synthesis...
        FM Synthesis completed!
        Kernel execution time: 13.146 ms
        WAV file written: output_24bit_48kHz.wav
        Memory Usage: 0 megabytes
```

### Manually Convert .wav to .mp3
```bash
ffmpeg -i output_32bit_48kHz.wav -vn -ar 44100 -ac 2 -b:a 192k output_demo.mp3
```

#### Example Audio Output

![Audio](https://github.com/Avicted/HIP_FM_Synthesis/blob/main/output_demo.mp3)

## License
This project is licensed under the MIT License