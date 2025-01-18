# Installation

## Prerequisites

- **Python 3.9 or higher**: Ensure Python is installed on your system.
- **FFmpeg**: Required for audio file handling.

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

## Install the Package

You can install the package via pip:

```bash
pip install live_audio_capture
```

---

## Verify Installation

To verify that the package is installed correctly, run the following command:

```bash
pip show live_audio_capture
```

---

## Troubleshooting

- **ModuleNotFoundError**: If you encounter a `ModuleNotFoundError`, ensure that the package is installed in the correct Python environment.
- **FFmpeg Not Found**: If FFmpeg is not found, ensure it is installed and added to your system's `PATH`.