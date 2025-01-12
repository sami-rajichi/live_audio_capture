import sys
import numpy as np
import subprocess
import platform
from typing import Generator, List, Optional
from .exceptions import UnsupportedPlatformError, UnsupportedAudioFormatError
from .mic_detection import get_default_mic
from .vad import VoiceActivityDetector
from scipy.io.wavfile import write as write_wav

class LiveAudioCapture:
    """
    A cross-platform utility for capturing live audio from a microphone using FFmpeg.
    Features:
    - Continuous listening mode.
    - Dynamic recording based on voice activity.
    - Silence duration threshold for stopping recording.
    - Optional beep sounds for start/stop feedback.
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        chunk_duration: float = 0.1,
        audio_format: str = "f32le",
        channels: int = 1,
        vad_threshold: float = 0.02,
        noise_floor_alpha: float = 0.9,
        hysteresis_high: float = 1.5,
        hysteresis_low: float = 0.5,
        enable_beep: bool = True,
    ):
        """
        Initialize the LiveAudioCapture instance.

        Args:
            sampling_rate (int): Sample rate in Hz (e.g., 16000).
            chunk_duration (float): Duration of each audio chunk in seconds (e.g., 0.1).
            audio_format (str): Audio format for FFmpeg output (e.g., "f32le").
            channels (int): Number of audio channels (1 for mono, 2 for stereo).
            vad_threshold (float): Initial energy threshold for speech detection.
            noise_floor_alpha (float): Smoothing factor for noise floor estimation.
            hysteresis_high (float): Multiplier for the threshold when speech is detected.
            hysteresis_low (float): Multiplier for the threshold when speech is not detected.
            enable_beep (bool): Whether to play beep sounds when recording starts/stops.
        """
        self.sampling_rate = sampling_rate
        self.chunk_duration = chunk_duration
        self.audio_format = audio_format
        self.channels = channels
        self.enable_beep = enable_beep
        self.process: Optional[subprocess.Popen] = None

        # Initialize VAD
        self.vad = VoiceActivityDetector(
            sample_rate=sampling_rate,
            frame_duration=chunk_duration,
            initial_threshold=vad_threshold,
            noise_floor_alpha=noise_floor_alpha,
            hysteresis_high=hysteresis_high,
            hysteresis_low=hysteresis_low,
        )

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

    def _play_beep(self, frequency: int, duration: int) -> None:
        """
        Play a beep sound with the specified frequency and duration.

        Args:
            frequency (int): Frequency of the beep sound in Hz.
            duration (int): Duration of the beep sound in milliseconds.
        """
        if not self.enable_beep:
            return

        if platform.system() == "Windows":
            # Use winsound for Windows
            import winsound
            winsound.Beep(frequency, duration)
        elif platform.system() == "Darwin":  # macOS
            # Use afplay for macOS (generate a sine wave using sox)
            import os
            os.system(f'play -nq -t alsa synth {duration/1000} sine {frequency}')
        else:
            # Use beep for Linux (requires 'beep' package)
            import os
            os.system(f'beep -f {frequency} -l {duration}')

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

        except KeyboardInterrupt:
            print("\nRecording interrupted by user.")
        finally:
            self.stop()

    def listen_and_record_with_vad(self, output_file: str = "output.wav", silence_duration: float = 2.0) -> None:
        """
        Continuously listen to the microphone and record speech segments.

        Args:
            output_file (str): Path to save the recorded audio file.
            silence_duration (float): Duration of silence (in seconds) to stop recording.
        """
        speech_segments: List[np.ndarray] = []
        recording = False
        silent_frames = 0
        silence_threshold_frames = int(silence_duration / self.chunk_duration)

        try:
            for audio_chunk in self.stream_audio():
                # Process the audio chunk with VAD
                is_speech = self.vad.process_audio(audio_chunk)

                if is_speech:
                    # Speech detected
                    if not recording:
                        print("Starting recording...")
                        recording = True
                        self._play_beep(1000, 200)  # High-pitched beep for start
                    speech_segments.append(audio_chunk)
                    silent_frames = 0  # Reset silence counter
                else:
                    # Silence detected
                    if recording:
                        silent_frames += 1
                        if silent_frames >= silence_threshold_frames:
                            # Stop recording if silence exceeds the threshold
                            print("Stopping recording due to silence.")
                            recording = False
                            self._play_beep(500, 200)  # Low-pitched beep for stop

                            # Save the recorded speech segment
                            if speech_segments:
                                combined_audio = np.concatenate(speech_segments)
                                write_wav(output_file, self.sampling_rate, (combined_audio * 32767).astype(np.int16))
                                print(f"Recording saved to {output_file}")
                                speech_segments = []  # Reset for the next segment
                        else:
                            # Add silence to the current recording
                            speech_segments.append(audio_chunk)

        except KeyboardInterrupt:
            print("\nContinuous listening interrupted by user.")

        # Save any remaining speech segments
        if speech_segments:
            combined_audio = np.concatenate(speech_segments)
            write_wav(output_file, self.sampling_rate, (combined_audio * 32767).astype(np.int16))
            print(f"Final recording saved to {output_file}")

    def stop(self):
        """Stop the FFmpeg process."""
        if self.process:
            self.process.terminate()
            self.process.wait()