# Live Audio Capture

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![PyPI Version](https://img.shields.io/pypi/v/live_audio_capture)

**Live Audio Capture** is a cross-platform Python package designed for capturing, processing, and analyzing live audio from a microphone in real-time. It provides a robust and flexible interface for voice activity detection (VAD), noise reduction, audio visualization, and more. Whether you're building a voice assistant, a transcription tool, or a real-time audio analysis application, this package has you covered.

---

## Why Use Live Audio Capture?

### Key Advantages
1. **Cross-Platform Support**: Works seamlessly on Windows, macOS, and Linux.
2. **Real-Time Processing**: Captures and processes audio in real-time with minimal latency.
3. **Voice Activity Detection (VAD)**: Dynamically detects speech and stops recording during silence.
4. **Noise Reduction**: Advanced noise reduction algorithms powered by the `noisereduce` package for cleaner audio.
5. **Customizable**: Highly configurable parameters for sampling rate, chunk duration, noise reduction, and more.
6. **Real-Time Visualization**: Visualize audio waveforms, frequency spectra, and spectrograms in real-time.
7. **Easy to Use**: Simple API for quick integration into your projects.

---

## Use Cases
- **Voice Assistants**: Capture and process user commands in real-time.
- **Transcription Tools**: Record and transcribe audio with noise reduction.
- **Real-Time Audio Analysis**: Analyze audio signals for frequency, volume, and other metrics.
- **Educational Tools**: Teach audio processing and visualization concepts.
- **Security Systems**: Detect and record audio events in real-time.

---

## Features
- **Live Audio Capture**: Capture audio from the microphone in real-time.
- **Voice Activity Detection (VAD)**: Automatically detect speech and stop recording during silence.
- **Noise Reduction**: Reduce background noise using the `noisereduce` package, which employs spectral gating techniques.
- **Real-Time Visualization**: Visualize audio waveforms, frequency spectra, and spectrograms.
- **Multiple Output Formats**: Save recordings in WAV, MP3, or OGG formats.
- **Customizable Parameters**:
  - Sampling rate
  - Chunk duration
  - VAD aggressiveness
  - Noise reduction settings
  - Low-pass filter cutoff frequency
- **Cross-Platform**: Works on Windows, macOS, and Linux.

---

## Installation

### Requirements
- Python 3.9 or higher
- FFmpeg (for audio file handling)
- Microphone access

### Install the Package
You can install the package via pip:

```bash
pip install live_audio_capture
```

### Install FFmpeg
- **Linux**:
  ```bash
  sudo apt update
  sudo apt install ffmpeg
  ```
- **macOS** (using Homebrew):
  ```bash
  brew install ffmpeg
  ```
- **Windows**: Download FFmpeg from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) and add it to your system's `PATH`.

---

## Usage

### Basic Example
Capture audio with voice activity detection and save it to a file:

```python
from live_audio_capture import LiveAudioCapture

# Initialize the audio capture
capture = LiveAudioCapture(
    sampling_rate=16000,  # Sample rate in Hz
    chunk_duration=0.1,   # Duration of each audio chunk in seconds
    enable_noise_canceling=True,  # Enable noise reduction
    aggressiveness=2,     # VAD aggressiveness level (0-3)
)

# Start recording with VAD
capture.listen_and_record_with_vad(
    output_file="output.wav",  # Save the recording to this file
    silence_duration=2.0,      # Stop recording after 2 seconds of silence
    format="wav",              # Output format
)

# Stop the capture
capture.stop()
```

### Real-Time Visualization
Visualize audio in real-time:

```python
from live_audio_capture import LiveAudioCapture, AudioVisualizer

# Initialize the audio capture
capture = LiveAudioCapture(sampling_rate=44100, chunk_duration=0.1)

# Initialize the audio visualizer
visualizer = AudioVisualizer(sampling_rate=44100, chunk_duration=0.1)

# Stream audio and visualize it
for audio_chunk in capture.stream_audio():
    visualizer.add_audio_chunk(audio_chunk)
```

### Advanced Example
Use all available parameters for maximum customization:

```python
from live_audio_capture import LiveAudioCapture

# Initialize the audio capture with all parameters
capture = LiveAudioCapture(
    sampling_rate=16000,
    chunk_duration=0.1,
    audio_format="f32le",
    channels=1,
    aggressiveness=3,
    enable_beep=True,
    enable_noise_canceling=True,
    low_pass_cutoff=7500.0,
    stationary_noise_reduction=True,
    prop_decrease=1.0,
    n_std_thresh_stationary=1.5,
    n_jobs=1,
    use_torch=False,
    device="cpu",
    calibration_duration=2.0,
    use_adaptive_threshold=True,
)

# Start recording with VAD
capture.listen_and_record_with_vad(
    output_file="output.wav",
    silence_duration=2.0,
    format="wav",
)

# Stop the capture
capture.stop()
```

---

## Features and Arguments

### `LiveAudioCapture` Parameters
- **`sampling_rate`**: Sample rate in Hz (default: `16000`).
- **`chunk_duration`**: Duration of each audio chunk in seconds (default: `0.1`).
- **`audio_format`**: Audio format for FFmpeg output (default: `"f32le"`).
- **`channels`**: Number of audio channels (default: `1` for mono).
- **`aggressiveness`**: VAD aggressiveness level (0-3, default: `1`).
- **`enable_beep`**: Play beep sounds when recording starts/stops (default: `True`).
- **`enable_noise_canceling`**: Enable noise reduction using the `noisereduce` package (default: `False`).
- **`low_pass_cutoff`**: Low-pass filter cutoff frequency (default: `7500.0`).
- **`stationary_noise_reduction`**: Enable stationary noise reduction (default: `False`).
- **`prop_decrease`**: Proportion to reduce noise by (default: `1.0`).
- **`n_std_thresh_stationary`**: Threshold for stationary noise reduction (default: `1.5`).
- **`n_jobs`**: Number of parallel jobs for noise reduction (default: `1`).
- **`use_torch`**: Use PyTorch for noise reduction (default: `False`).
- **`device`**: Device for PyTorch noise reduction (default: `"cpu"`).
- **`calibration_duration`**: Duration of calibration for adaptive thresholding (default: `2.0`).
- **`use_adaptive_threshold`**: Enable adaptive thresholding for VAD (default: `True`).

---

## Recommendations
1. **Use Threading for Real-Time Listening**: It is highly recommended to use threading for real-time audio listening. This allows you to easily stop the audio capture in any script using the `.stop()` method without blocking the main program.
2. **Use a High-Quality Microphone**: For best results, use a microphone with good noise cancellation.
3. **Adjust VAD Aggressiveness**: Higher aggressiveness levels may reduce false positives but can also miss softer speech.
4. **Enable Noise Reduction**: If you're working in a noisy environment, enable noise reduction for cleaner audio.
5. **Test on Your Platform**: Test the package on your target platform to ensure compatibility.

---

## Technical Details

### Voice Activity Detection (VAD)
The VAD system uses an energy-based approach with adaptive thresholding. It calculates the energy of each audio chunk and compares it to a dynamically adjusted threshold. Hysteresis is applied to avoid rapid toggling between speech and silence states.

### Noise Reduction
The `noisereduce` package is used for noise reduction. It employs spectral gating techniques to remove background noise while preserving speech. You can choose between stationary and non-stationary noise reduction, and even use PyTorch for GPU-accelerated processing.

### Real-Time Visualization
The visualization module provides insights into:
- **Waveform**: The amplitude of the audio signal over time.
- **Frequency Spectrum**: The distribution of frequencies in the audio signal.
- **Spectrogram**: A visual representation of the spectrum of frequencies over time.
- **Volume Meter**: Real-time volume levels.
- **Volume History**: A history of volume levels over time.

---

## Contributing
Contributions are welcome! Please read the [Contributing Guidelines](CONTRIBUTING.md) for details.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Support
For questions, issues, or feature requests, please open an issue on [GitHub](https://github.com/sami-rajichi/live_audio_capture/issues).

---

## Final Words
If you find this package useful, please consider leaving a ⭐ star on the [GitHub repository](https://github.com/sami-rajichi/live_audio_capture). Your support motivates us to keep improving! If you have any suggestions for optimization or new features, don't hesitate to reach out. We'd love to hear from you!
