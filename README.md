# FM Synthesizer with HIP

This project demonstrates **FM Synthesis** (Frequency Modulation) using **HIP** (Heterogeneous Compute), enabling high-performance sound generation on both AMD and NVIDIA GPUs.

## Key Features
FM Synthesis:

 - Implements basic **FM synthesis** to generate sounds by modulating one oscillator's frequency (carrier) using another oscillator (modulator).

 - **Dynamic Frequency Modulation**: Both carrier and modulator frequencies vary over time.

 - Audio data is generated and then written to a .wav file for playback or further processing.

 - Supports 16, 24 and 32 -bit PCM Output at variable sample rates.

 - Multiple waveforms supported: Sine, Square, Triangle, Sawtooth.

## Dependencies
 - Ensure that your system supports HIP (either via AMD ROCm or via NVIDIA CUDA).
 - Make

## Build and run
```bash
make
```

## License
This project is licensed under the MIT License