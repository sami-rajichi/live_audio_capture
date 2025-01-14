import sys
import numpy as np
import subprocess
from typing import Generator, Optional, List
from .vad import VoiceActivityDetector
from .exceptions import UnsupportedPlatformError, UnsupportedAudioFormatError
from .mic_detection import get_default_mic
from .audio_processing import apply_noise_reduction, apply_low_pass_filter
from scipy.io.wavfile import write as write_wav
from pydub import AudioSegment

class LiveAudioCapture:
    """
    A cross-platform utility for capturing live audio from a microphone using FFmpeg.
    Features:
    - Continuous listening mode.
    - Dynamic recording based on voice activity.
    - Silence duration threshold for stopping recording.
    - Optional beep sounds for start/stop feedback.
    - Save recordings in multiple formats (WAV, MP3, OGG).
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
        noise_threshold_db: float = -50.0,
        low_pass_cutoff: float = 7800.0
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
            noise_threshold_db (float): Noise threshold in dB for noise cancellation.
            low_pass_cutoff (float): Cutoff frequency for the low-pass filter.
        """
        self.sampling_rate = sampling_rate
        self.chunk_duration = chunk_duration
        self.audio_format = audio_format
        self.channels = channels
        self.enable_beep = enable_beep
        self.noise_threshold_db = noise_threshold_db
        self.low_pass_cutoff = low_pass_cutoff
        self.process: Optional[subprocess.Popen] = None
        self.is_streaming = False  # Flag to control streaming
        self.is_recording = False  # Flag to control recording
        
        # Validate the cutoff frequency
        nyquist = 0.5 * self.sampling_rate
        if self.low_pass_cutoff >= nyquist:
            raise ValueError(
                f"Cutoff frequency must be less than the Nyquist frequency ({nyquist} Hz). "
                f"Provided cutoff frequency: {self.low_pass_cutoff} Hz."
            )

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

        if sys.platform == "win32":
            # Use winsound for Windows
            import winsound
            winsound.Beep(frequency, duration)
        elif sys.platform == "darwin":  # macOS
            # Use afplay for macOS (generate a sine wave using sox)
            import os
            os.system(f'play -nq -t alsa synth {duration/1000} sine {frequency}')
        elif sys.platform == "linux":
            # Use beep for Linux (requires 'beep' package)
            import os
            os.system(f'beep -f {frequency} -l {duration}')
        else:
            raise UnsupportedPlatformError(f"Unsupported platform: {sys.platform}")

    def stream_audio(self) -> Generator[np.ndarray, None, None]:
        """Stream live audio from the microphone."""
        self._start_ffmpeg_process()
        self.is_streaming = True

        try:
            while self.is_streaming:
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
            print("\nStreaming interrupted by user.")
        finally:
            self.stop_streaming()

    def stop_streaming(self) -> None:
        """Stop the audio stream and terminate the FFmpeg process."""
        self.is_streaming = False
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
        print("Streaming stopped.")

    def save_recording(self, audio_data: np.ndarray, output_file: str, format: str = "wav") -> None:
        """
        Save the recorded audio to a file in the specified format.

        Args:
            audio_data (np.ndarray): The recorded audio data.
            output_file (str): Path to save the recorded audio file.
            format (str): Output format (e.g., "wav", "mp3", "ogg").
        """
        # Scale the audio data to the appropriate range
        if self.audio_format == "f32le":
            # Scale floating-point data to the range [-1, 1]
            audio_data = np.clip(audio_data, -1.0, 1.0)
            # Convert to 16-bit integer format for saving
            audio_data = (audio_data * 32767).astype(np.int16)
        elif self.audio_format == "s16le":
            # Data is already in 16-bit integer format
            audio_data = audio_data.astype(np.int16)
        else:
            raise UnsupportedAudioFormatError(f"Unsupported audio format: {self.audio_format}")

        # Convert the NumPy array to a PyDub AudioSegment
        audio_segment = AudioSegment(
            audio_data.tobytes(),
            frame_rate=self.sampling_rate,
            sample_width=2,  # 16-bit audio (2 bytes per sample)
            channels=self.channels,
        )

        # Normalize the volume to prevent clipping
        audio_segment = audio_segment.normalize()

        # Save the audio in the specified format
        audio_segment.export(output_file, format=format)
        print(f"Recording saved to {output_file} in {format.upper()} format.")

    def process_audio_chunk(self, audio_chunk: np.ndarray, enable_noise_canceling: bool = True) -> np.ndarray:
        """
        Process an audio chunk with optional noise cancellation.

        Args:
            audio_chunk (np.ndarray): The audio chunk to process.
            enable_noise_canceling (bool): Whether to apply noise cancellation.

        Returns:
            np.ndarray: The processed audio chunk.
        """
        if enable_noise_canceling:
            # Apply noise reduction
            audio_chunk = apply_noise_reduction(audio_chunk, self.sampling_rate, self.noise_threshold_db)
            # Apply low-pass filter
            audio_chunk = apply_low_pass_filter(audio_chunk, self.sampling_rate, self.low_pass_cutoff)
        return audio_chunk

    def listen_and_record_with_vad(
        self,
        output_file: str = "output.wav",
        silence_duration: float = 2.0,
        format: str = "wav",
        enable_noise_canceling: bool = True,
    ) -> None:
        """
        Continuously listen to the microphone and record speech segments with optional noise cancellation.

        Args:
            output_file (str): Path to save the recorded audio file.
            silence_duration (float): Duration of silence (in seconds) to stop recording.
            format (str): Output format (e.g., "wav", "mp3", "ogg").
            enable_noise_canceling (bool): Whether to apply noise cancellation to the audio chunks.
        """
        speech_segments: List[np.ndarray] = []
        self.is_recording = False
        silent_frames = 0
        silence_threshold_frames = int(silence_duration / self.chunk_duration)

        try:
            for audio_chunk in self.stream_audio():
                # Process the audio chunk with optional noise cancellation
                processed_chunk = self.process_audio_chunk(audio_chunk, enable_noise_canceling)

                # Process the audio chunk with VAD
                is_speech = self.vad.process_audio(processed_chunk)

                if is_speech:
                    # Speech detected
                    if not self.is_recording:
                        print("Starting recording...")
                        self.is_recording = True
                        self._play_beep(1000, 200)  # High-pitched beep for start
                    speech_segments.append(processed_chunk)
                    silent_frames = 0  # Reset silence counter
                else:
                    # Silence detected
                    if self.is_recording:
                        silent_frames += 1
                        if silent_frames >= silence_threshold_frames:
                            # Stop recording if silence exceeds the threshold
                            print("Stopping recording due to silence.")
                            self.is_recording = False
                            self._play_beep(500, 200)  # Low-pitched beep for stop

                            # Save the recorded speech segment
                            if speech_segments:
                                combined_audio = np.concatenate(speech_segments)
                                self.save_recording(combined_audio, output_file, format=format)
                                speech_segments = []  # Reset for the next segment
                        else:
                            # Add silence to the current recording
                            speech_segments.append(processed_chunk)

        except KeyboardInterrupt:
            print("\nContinuous listening interrupted by user.")

        # Save any remaining speech segments
        if speech_segments:
            combined_audio = np.concatenate(speech_segments)
            self.save_recording(combined_audio, output_file, format=format)

    def stop_recording(self) -> None:
        """Stop the recording process."""
        self.is_recording = False
        print("Recording stopped.")

    def stop(self):
        """Stop both streaming and recording."""
        self.stop_streaming()
        self.stop_recording()