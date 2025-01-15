import sys
import time
import numpy as np
import subprocess
from typing import Generator, Optional, List
from .vad import VoiceActivityDetector
from .exceptions import UnsupportedPlatformError, UnsupportedAudioFormatError
from .mic_detection import get_default_mic
from .audio_processing import apply_noise_reduction, apply_low_pass_filter
from scipy.io.wavfile import write as write_wav
from pydub import AudioSegment
import simpleaudio as sa

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
        aggressiveness: int = 1,  # Aggressiveness level for VAD
        enable_beep: bool = True,
        enable_noise_canceling: bool = False,
        low_pass_cutoff: float = 7500.0,
        stationary_noise_reduction: bool = False,
        prop_decrease: float = 1.0,
        n_std_thresh_stationary: float = 1.5,
        n_jobs: int = 1,
        use_torch: bool = False,
        device: str = "cuda",
    ):
        """
        Initialize the LiveAudioCapture instance.

        Args:
            sampling_rate (int): Sample rate in Hz (e.g., 16000).
            chunk_duration (float): Duration of each audio chunk in seconds (e.g., 0.1).
            audio_format (str): Audio format for FFmpeg output (e.g., "f32le").
            channels (int): Number of audio channels (1 for mono, 2 for stereo).
            aggressiveness (int): Aggressiveness level for VAD (0 = least aggressive, 3 = most aggressive).
            enable_beep (bool): Whether to play beep sounds when recording starts/stops.
            enable_noise_canceling (bool): Whether to apply noise cancellation.
            low_pass_cutoff (float): Cutoff frequency for the low-pass filter.
            stationary_noise_reduction (bool): Whether to use stationary noise reduction.
            prop_decrease (float): Proportion to reduce noise by (1.0 = 100%).
            n_std_thresh_stationary (float): Threshold for stationary noise reduction.
            n_jobs (int): Number of parallel jobs to run. Set to -1 to use all CPU cores.
            use_torch (bool): Whether to use the PyTorch version of spectral gating.
            device (str): Device to run the PyTorch spectral gating on (e.g., "cuda" or "cpu").
        """
        self.sampling_rate = sampling_rate
        self.chunk_duration = chunk_duration
        self.audio_format = audio_format
        self.channels = channels
        self.enable_beep = enable_beep
        self.enable_noise_canceling = enable_noise_canceling
        self.low_pass_cutoff = low_pass_cutoff
        self.stationary_noise_reduction = stationary_noise_reduction
        self.prop_decrease = prop_decrease
        self.n_std_thresh_stationary = n_std_thresh_stationary
        self.n_jobs = n_jobs
        self.use_torch = use_torch
        self.device = device
        self.process: Optional[subprocess.Popen] = None
        self.is_streaming = False
        self.is_recording = False

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
            aggressiveness=aggressiveness,
            hysteresis_high=1.5,
            hysteresis_low=0.5,
            enable_noise_canceling=self.enable_noise_canceling
        )

        # Determine the input device based on the platform
        if sys.platform == "linux":
            self.input_format = "alsa"
        elif sys.platform == "darwin":  # macOS
            self.input_format = "avfoundation"
        elif sys.platform == "win32":
            self.input_format = "dshow"
        else:
            raise RuntimeError(f"Unsupported platform: {sys.platform}")

        # Get the default microphone
        self.input_device = get_default_mic()
        print(f"Using input device: {self.input_device}")

    def _start_ffmpeg_process(self) -> None:
        """Start the FFmpeg process for capturing live audio."""
        if self.process is not None:
            return  # FFmpeg process is already running

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

        try:
            # Start FFmpeg process
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            raise RuntimeError(f"Failed to start FFmpeg process: {e}")

    def _play_beep(self, frequency: int, duration: int) -> None:
        """
        Play a beep sound asynchronously using the simpleaudio library.

        Args:
            frequency (int): Frequency of the beep sound in Hz.
            duration (int): Duration of the beep sound in milliseconds.
        """
        if not self.enable_beep:
            return

        # Generate a sine wave for the beep sound
        sample_rate = 44100  # Standard sample rate for audio playback
        t = np.linspace(0, duration / 1000, int(sample_rate * duration / 1000), endpoint=False)
        waveform = np.sin(2 * np.pi * frequency * t)
        waveform = (waveform * 32767).astype(np.int16)  # Convert to 16-bit PCM format

        # Play the sound asynchronously
        play_obj = sa.play_buffer(waveform, num_channels=1, bytes_per_sample=2, sample_rate=sample_rate)
        play_obj.stop()  # Ensure the sound stops after playing

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
                    raise RuntimeError(f"Unsupported audio format: {self.audio_format}")

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
            try:
                self.process.terminate()
                self.process.wait(timeout=5)  # Wait for the process to terminate
            except subprocess.TimeoutExpired:
                self.process.kill()  # Force kill if it doesn't terminate
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
            raise RuntimeError(f"Unsupported audio format: {self.audio_format}")

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
            # Apply noise reduction using noisereduce
            audio_chunk = apply_noise_reduction(
                audio_chunk,
                self.sampling_rate,
                stationary=self.stationary_noise_reduction,
                prop_decrease=self.prop_decrease,
                n_std_thresh_stationary=self.n_std_thresh_stationary,
                n_jobs=self.n_jobs,  # Pass the number of parallel jobs
                use_torch=self.use_torch,  # Enable/disable PyTorch
                device=self.device,  # Specify the device for PyTorch
            )
            # Apply low-pass filter
            audio_chunk = apply_low_pass_filter(audio_chunk, self.sampling_rate, self.low_pass_cutoff)
        return audio_chunk

    def listen_and_record_with_vad(
        self,
        output_file: str = "output.wav",
        silence_duration: float = 2.0,
        format: str = "wav"
    ) -> None:
        """
        Continuously listen to the microphone and record speech segments.

        Args:
            output_file (str): Path to save the recorded audio file.
            silence_duration (float): Duration of silence (in seconds) to stop recording.
            format (str): Output format (e.g., "wav", "mp3", "ogg").
        """
        speech_segments: List[np.ndarray] = []
        self.is_recording = False
        silent_frames = 0
        silence_threshold_frames = int(silence_duration / self.chunk_duration)

        try:
            for audio_chunk in self.stream_audio():
                # Process the audio chunk with optional noise cancellation
                processed_chunk = self.process_audio_chunk(audio_chunk, self.enable_noise_canceling)

                # Process the audio chunk with VAD
                is_speech = self.vad.process_audio(processed_chunk)

                if is_speech:
                    # Speech detected
                    if not self.is_recording:
                        print("\nStarting recording...")
                        self.is_recording = True
                        self._play_beep(600, 200)  # High-pitched beep for start
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
                            self._play_beep(300, 200)  # Low-pitched beep for stop (async)

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