import numpy as np

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
        enable_noise_canceling: bool = False
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
        """
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration)
        self.aggressiveness = aggressiveness
        self.hysteresis_high = hysteresis_high
        self.hysteresis_low = hysteresis_low
        self.enable_noise_canceling = enable_noise_canceling

        # Set initial threshold based on aggressiveness
        self.initial_threshold = self._get_initial_threshold(aggressiveness)
        self.current_threshold = self.initial_threshold
        self.speech_active = False

        # Debugging: Print initial settings
        print(f"Initialized VAD with aggressiveness={aggressiveness}, initial_threshold={self.initial_threshold:.6f}")

    def _get_initial_threshold(self, aggressiveness: int) -> float:
        """
        Get the initial energy threshold based on the aggressiveness level.

        Args:
            aggressiveness (int): Aggressiveness level (0, 1, 2, or 3).

        Returns:
            float: Initial energy threshold.
        """
        # Adjust these values based on your requirements
        if aggressiveness == 0:
            return 0.0005 if not self.enable_noise_canceling else 0.00002 # Least aggressive (lowest threshold)
        elif aggressiveness == 1:
            return 0.001 if not self.enable_noise_canceling else 0.00003
        elif aggressiveness == 2:
            return 0.002 if not self.enable_noise_canceling else 0.00004
        elif aggressiveness == 3:
            return 0.005 if not self.enable_noise_canceling else 0.0001 # Most aggressive (highest threshold)
        else:
            raise ValueError("Aggressiveness must be between 0 and 3.")

    def _calculate_energy(self, frame: np.ndarray) -> float:
        """
        Calculate the energy of a single audio frame.

        Args:
            frame (np.ndarray): Audio frame.

        Returns:
            float: Energy of the frame.
        """
        return np.sum(frame**2) / len(frame)

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

    def process_audio(self, audio_chunk: np.ndarray) -> bool:
        """
        Process an audio chunk and determine if speech is detected.

        Args:
            audio_chunk (np.ndarray): Audio chunk to process.

        Returns:
            bool: True if speech is detected, False otherwise.
        """
        # Calculate energy
        energy = self._calculate_energy(audio_chunk)

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