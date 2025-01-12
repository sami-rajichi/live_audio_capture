import sys
import numpy as np
import subprocess
from typing import Generator, Optional
from .exceptions import UnsupportedPlatformError, UnsupportedAudioFormatError
from .mic_detection import get_default_mic

class LiveAudioCapture:
    """
    A cross-platform utility for capturing live audio from a microphone using FFmpeg.
    """

    def __init__(self, sampling_rate: int = 16000, chunk_duration: float = 0.1, audio_format: str = "f32le", channels: int = 1):
        """
        Initialize the LiveAudioCapture instance.

        Args:
            sampling_rate (int): Sample rate in Hz (e.g., 16000).
            chunk_duration (float): Duration of each audio chunk in seconds (e.g., 0.1).
            audio_format (str): Audio format for FFmpeg output (e.g., "f32le").
            channels (int): Number of audio channels (1 for mono, 2 for stereo).
        """
        self.sampling_rate = sampling_rate
        self.chunk_duration = chunk_duration
        self.audio_format = audio_format
        self.channels = channels
        self.process: Optional[subprocess.Popen] = None

        # Determine the input device based on the platform
        if sys.platform == "linux":
            self.input_format = "alsa"
        elif sys.platform == "darwin":  # macOS
            self.input_format = "avfoundation"
        elif sys.platform == "win32":
            self.input_format = "dshow"
        else:
            raise UnsupportedPlatformError(f"Unsupported platform: {sys.platform}")

        # Get the default microphone
        self.input_device = get_default_mic()
        print(f"Using input device: {self.input_device}")

    def _start_ffmpeg_process(self) -> None:
        """Start the FFmpeg process for capturing live audio."""
        # Calculate chunk size in bytes
        bytes_per_sample = 4 if self.audio_format == "f32le" else 2  # 32-bit float or 16-bit int
        self.chunk_size = int(self.sampling_rate * self.chunk_duration * self.channels * bytes_per_sample)

        # FFmpeg command to capture live audio
        command = [
            "ffmpeg",
            "-f", self.input_format,       # Input format (platform-specific)
            "-i", self.input_device,       # Input device (platform-specific)
            "-ar", str(self.sampling_rate),  # Sample rate
            "-ac", str(self.channels),     # Number of channels
            "-f", self.audio_format,      # Output format
            "-"                           # Output to stdout
        ]

        # Start FFmpeg process
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def stream_audio(self) -> Generator[np.ndarray, None, None]:
        """Stream live audio from the microphone."""
        self._start_ffmpeg_process()

        try:
            while True:
                # Read raw audio data from FFmpeg stdout
                raw_data = self.process.stdout.read(self.chunk_size)
                if not raw_data:
                    # Print FFmpeg errors if no data is received
                    stderr = self.process.stderr.read().decode()
                    if stderr:
                        print("FFmpeg errors:", stderr)
                    break

                # Convert raw data to NumPy array
                if self.audio_format == "f32le":
                    audio_chunk = np.frombuffer(raw_data, dtype=np.float32)
                elif self.audio_format == "s16le":
                    audio_chunk = np.frombuffer(raw_data, dtype=np.int16) / 32768.0  # Normalize to [-1, 1]
                else:
                    raise UnsupportedAudioFormatError(f"Unsupported audio format: {self.audio_format}")

                # Yield the audio chunk for processing
                yield audio_chunk

        finally:
            self.stop()

    def stop(self):
        """Stop the FFmpeg process."""
        if self.process:
            self.process.terminate()
            self.process.wait()
