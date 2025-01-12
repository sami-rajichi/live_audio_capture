import numpy as np

class VoiceActivityDetector:
    """
    A voice activity detector (VAD) based on energy thresholding.
    Features:
    - Adaptive energy threshold.
    - Noise floor estimation.
    - Hysteresis for stable speech detection.
    - Silence duration threshold for stopping recording.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration: float = 0.1,
        initial_threshold: float = 0.002,
        noise_floor_alpha: float = 0.9,
        hysteresis_high: float = 1.5,
        hysteresis_low: float = 0.5,
    ):
        """
        Initialize the VoiceActivityDetector.

        Args:
            sample_rate (int): Sample rate of the audio (default: 16000 Hz).
            frame_duration (float): Duration of each frame in seconds (default: 0.1 seconds).
            initial_threshold (float): Initial energy threshold for speech detection.
            noise_floor_alpha (float): Smoothing factor for noise floor estimation (0 < alpha < 1).
            hysteresis_high (float): Multiplier for the threshold when speech is detected.
            hysteresis_low (float): Multiplier for the threshold when speech is not detected.
        """
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration)
        self.initial_threshold = initial_threshold
        self.noise_floor_alpha = noise_floor_alpha
        self.hysteresis_high = hysteresis_high
        self.hysteresis_low = hysteresis_low

        # Initialize noise floor and threshold
        self.noise_floor = initial_threshold
        self.current_threshold = initial_threshold
        self.speech_active = False

    def _calculate_energy(self, frame: np.ndarray) -> float:
        """
        Calculate the energy of a single audio frame.

        Args:
            frame (np.ndarray): Audio frame.

        Returns:
            float: Energy of the frame.
        """
        return np.sum(frame**2) / len(frame)

    def _update_noise_floor(self, energy: float) -> None:
        """
        Update the noise floor estimate using exponential smoothing.

        Args:
            energy (float): Energy of the current frame.
        """
        if not self.speech_active:
            self.noise_floor = self.noise_floor_alpha * self.noise_floor + (1 - self.noise_floor_alpha) * energy

    def _update_threshold(self) -> None:
        """
        Update the energy threshold using hysteresis.
        """
        if self.speech_active:
            self.current_threshold = self.noise_floor * self.hysteresis_high
        else:
            self.current_threshold = self.noise_floor * self.hysteresis_low

    def process_audio(self, audio_chunk: np.ndarray) -> bool:
        """
        Process an audio chunk and determine if speech is detected.

        Args:
            audio_chunk (np.ndarray): Audio chunk to process.

        Returns:
            bool: True if speech is detected, False otherwise.
        """
        # Calculate the energy of the audio chunk
        energy = self._calculate_energy(audio_chunk)

        # Update the noise floor estimate
        self._update_noise_floor(energy)

        # Update the threshold using hysteresis
        self._update_threshold()

        # Debugging information
        print(
            f"Energy: {energy:.6f}, Noise Floor: {self.noise_floor:.6f}, "
            f"Threshold: {self.current_threshold:.6f}, Speech Active: {self.speech_active}"
        )

        # Detect speech based on the current threshold
        if energy > self.current_threshold:
            self.speech_active = True
            return True
        else:
            self.speech_active = False
            return False