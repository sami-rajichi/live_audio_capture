import numpy as np
import subprocess
import sys
from typing import List
from .audio_utils.audio_processing import AudioProcessing
from .audio_utils.mic_utils import MicUtils

class VoiceActivityDetector:
    """
    A simplified voice activity detector (VAD) similar to WebRTC VAD.
    Features:
    - Energy-based speech detection.
    - Aggressiveness level to control detection strictness.
    - Hysteresis for stable speech detection.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration: float = 0.03,
        aggressiveness: int = 1,  # Aggressiveness level (0, 1, 2, or 3)
        hysteresis_high: float = 1.5,
        hysteresis_low: float = 0.5,
        enable_noise_canceling: bool = False,
        calibration_duration: float = 2.0,  # Duration of calibration in seconds
        use_adaptive_threshold: bool = True,  # Enable adaptive thresholding
        audio_format: str = "f32le",  # Audio format for calibration
        channels: int = 1
    ):
        """
        Initialize the VoiceActivityDetector.

        Args:
            sample_rate (int): Sample rate of the audio (default: 16000 Hz).
            frame_duration (float): Duration of each frame in seconds (default: 0.03 seconds).
            aggressiveness (int): Aggressiveness level (0 = least aggressive, 3 = most aggressive).
            hysteresis_high (float): Multiplier for the threshold when speech is detected.
            hysteresis_low (float): Multiplier for the threshold when speech is not detected.
            enable_noise_canceling (bool): Whether to apply noise cancellation.
            calibration_duration (float): Duration of the calibration phase in seconds.
            use_adaptive_threshold (bool): Whether to use adaptive thresholding.
            audio_format (str): Audio format for calibration (e.g., "f32le" or "s16le").
            channels (int): Number of audio channels (1 for mono, 2 for stereo).
        """
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration)
        self.aggressiveness = aggressiveness
        self.hysteresis_high = hysteresis_high
        self.hysteresis_low = hysteresis_low
        self.enable_noise_canceling = enable_noise_canceling
        self.calibration_duration = calibration_duration
        self.use_adaptive_threshold = use_adaptive_threshold
        self.audio_format = audio_format
        self.channels = channels

        # Calibrate the initial threshold
        self.initial_threshold = self._calibrate_threshold() if self.use_adaptive_threshold else self._get_manual_threshold()
        self.current_threshold = self.initial_threshold
        self.speech_active = False

        print(f"Initialized VAD with aggressiveness={aggressiveness}, initial_threshold={self.initial_threshold:.6f}")

    def _calibrate_threshold(self) -> float:
        """
        Calibrate the initial energy threshold based on the background noise level.

        Returns:
            float: Calibrated initial energy threshold.
        """
        print("Calibrating threshold... Please remain silent for a few seconds.")
        audio_chunks = self._capture_calibration_audio()
        background_energy = np.mean([AudioProcessing.calculate_energy(chunk) for chunk in audio_chunks])
        print(f"Calibration complete. Background energy: {background_energy:.6f}")

        # Define multipliers based on aggressiveness
        multipliers = {
            0: 1.5,  # Least aggressive
            1: 2.0,
            2: 2.5,
            3: 3.0,  # Most aggressive
        }
        return background_energy * multipliers.get(self.aggressiveness, 2.0)

    def _capture_calibration_audio(self) -> List[np.ndarray]:
        """
        Capture a short audio sample for calibration.

        Returns:
            List[np.ndarray]: List of audio chunks captured during calibration.
        """
        # Start FFmpeg process for calibration
        command = [
            "ffmpeg",
            "-f", "alsa" if sys.platform == "linux" else "avfoundation" if sys.platform == "darwin" else "dshow",
            "-i", MicUtils.get_default_mic(),
            "-ar", str(self.sample_rate),
            "-ac", str(self.channels),
            "-f", self.audio_format,
            "-"
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Capture audio chunks for the calibration duration
        audio_chunks = []
        for _ in range(int(self.calibration_duration / self.frame_duration)):
            raw_data = process.stdout.read(self.frame_size * 4)  # 4 bytes per sample for f32le
            if not raw_data:
                break
            audio_chunk = AudioProcessing.process_audio_chunk(raw_data, self.audio_format)
            audio_chunks.append(audio_chunk)

        # Stop the FFmpeg process
        process.terminate()
        process.wait()

        return audio_chunks

    def _get_manual_threshold(self) -> float:
        """
        Get the initial energy threshold based on the aggressiveness level (manual values).

        Returns:
            float: Initial energy threshold.
        """
        # Manual thresholds
        if self.aggressiveness == 0:
            return 0.0005 if not self.enable_noise_canceling else 0.00002  # Least aggressive (lowest threshold)
        elif self.aggressiveness == 1:
            return 0.001 if not self.enable_noise_canceling else 0.00003
        elif self.aggressiveness == 2:
            return 0.002 if not self.enable_noise_canceling else 0.00004
        elif self.aggressiveness == 3:
            return 0.005 if not self.enable_noise_canceling else 0.0001  # Most aggressive (highest threshold)
        else:
            raise ValueError("Aggressiveness must be between 0 and 3.")

    def process_audio(self, audio_chunk: np.ndarray) -> bool:
        """
        Process an audio chunk and determine if speech is detected.

        Args:
            audio_chunk (np.ndarray): Audio chunk to process.

        Returns:
            bool: True if speech is detected, False otherwise.
        """
        # Calculate energy
        energy = AudioProcessing.calculate_energy(audio_chunk)

        # Detect speech based on energy
        if energy > self.current_threshold:
            self.speech_active = True
            print("Speech detected!")
        else:
            self.speech_active = False
            print("No speech detected.")

        # Update threshold using hysteresis
        self._update_threshold()

        # Debugging: Print key values
        print(
            f"Energy: {energy:.6f}, Current Threshold: {self.current_threshold:.6f}, "
            f"Speech Active: {self.speech_active}"
        )

        return self.speech_active

    def _update_threshold(self) -> None:
        """
        Update the energy threshold using hysteresis.
        """
        if self.speech_active:
            # Increase threshold slightly to avoid false positives
            self.current_threshold = self.initial_threshold * self.hysteresis_high
        else:
            # Lower threshold to detect speech more sensitively
            self.current_threshold = self.initial_threshold * self.hysteresis_low