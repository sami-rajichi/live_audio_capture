# Welcome to Live Audio Capture

**Live Audio Capture** is a cross-platform Python package designed for capturing, processing, and analyzing live audio from a microphone in real-time. Whether you're building a voice assistant, a transcription tool, or a real-time audio analysis application, this package has you covered.

---

## Features

- **Real-Time Audio Capture**: Capture audio from the microphone in real-time.
- **Voice Activity Detection (VAD)**: Automatically detect speech and stop recording during silence.
- **Noise Reduction**: Reduce background noise using advanced spectral gating techniques.
- **Real-Time Visualization**: Visualize audio waveforms, frequency spectra, and spectrograms.
- **Customizable**: Highly configurable parameters for sampling rate, chunk duration, noise reduction, and more.
- **Cross-Platform**: Works on Windows, macOS, and Linux.

---

## Quick Start

To get started, install the package:

```bash
pip install live_audio_capture
```

Then, capture audio with voice activity detection:

```python
from live_audio_capture import LiveAudioCapture

capture = LiveAudioCapture()
capture.listen_and_record_with_vad(output_file="output.wav")
```

---

## Explore the Documentation

- [Installation](installation.md): Learn how to install and set up the package.
- [Usage](usage.md): Discover how to use the package with examples.
- [API Reference](api.md): Explore the full API documentation.
- [Contributing](contributing.md): Find out how to contribute to the project.

---

## Support

If you have questions, issues, or feature requests, please [open an issue](https://github.com/sami-rajichi/live_audio_capture/issues) on GitHub.

---

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/sami-rajichi/live_audio_capture/blob/main/LICENSE) file for details.
