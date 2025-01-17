# live_audio_capture/audio_utils/audio_processing.py
import numpy as np
from pydub import AudioSegment
import noisereduce as nr

class AudioProcessing:
    """
    Utilities for processing audio data.
    """

    @staticmethod
    def calculate_energy(audio_chunk: np.ndarray) -> float:
        """
        Calculate the energy of an audio chunk.

        Args:
            audio_chunk (np.ndarray): The audio chunk to process.

        Returns:
            float: The energy of the audio chunk.
        """
        return np.sum(audio_chunk**2) / len(audio_chunk)

    @staticmethod
    def process_audio_chunk(raw_data: bytes, audio_format: str = "f32le") -> np.ndarray:
        """
        Convert raw audio data to a NumPy array based on the audio format.

        Args:
            raw_data (bytes): Raw audio data from the microphone.
            audio_format (str): Audio format (e.g., "f32le" or "s16le").

        Returns:
            np.ndarray: The processed audio chunk.
        """
        if audio_format == "f32le":
            return np.frombuffer(raw_data, dtype=np.float32)
        elif audio_format == "s16le":
            return np.frombuffer(raw_data, dtype=np.int16) / 32768.0  # Normalize to [-1, 1]
        else:
            raise ValueError(f"Unsupported audio format: {audio_format}")

    @staticmethod
    def apply_noise_reduction_to_file(
        input_file: str,
        output_file: str,
        stationary: bool = False,
        prop_decrease: float = 1.0,
        n_std_thresh_stationary: float = 1.5,
        n_jobs: int = 1,
        use_torch: bool = False,
        device: str = "cuda",
    ) -> None:
        """
        Apply noise reduction to an audio file and save the result.

        Args:
            input_file (str): Path to the input audio file.
            output_file (str): Path to save the processed audio file.
            stationary (bool): Whether to perform stationary noise reduction.
            prop_decrease (float): Proportion to reduce noise by (1.0 = 100%).
            n_std_thresh_stationary (float): Threshold for stationary noise reduction.
            n_jobs (int): Number of parallel jobs to run. Set to -1 to use all CPU cores.
            use_torch (bool): Whether to use the PyTorch version of spectral gating.
            device (str): Device to run the PyTorch spectral gating on (e.g., "cuda" or "cpu").
        """
        try:
            # Load the audio file using pydub
            audio = AudioSegment.from_file(input_file)

            # Convert the audio to a NumPy array
            samples = np.array(audio.get_array_of_samples())
            sample_rate = audio.frame_rate

            # Normalize the audio to the range [-1, 1]
            if audio.sample_width == 2:  # 16-bit audio
                samples = samples / 32768.0
            elif audio.sample_width == 4:  # 32-bit audio
                samples = samples / 2147483648.0

            # Apply noise reduction
            reduced_noise = nr.reduce_noise(
                y=samples,
                sr=sample_rate,
                stationary=stationary,
                prop_decrease=prop_decrease,
                n_std_thresh_stationary=n_std_thresh_stationary,
                n_jobs=n_jobs,
                use_torch=use_torch,
                device=device,
            )

            # Scale the audio back to the original range
            if audio.sample_width == 2:  # 16-bit audio
                reduced_noise = (reduced_noise * 32768.0).astype(np.int16)
            elif audio.sample_width == 4:  # 32-bit audio
                reduced_noise = (reduced_noise * 2147483648.0).astype(np.int32)

            # Convert the NumPy array back to an AudioSegment
            processed_audio = AudioSegment(
                reduced_noise.tobytes(),
                frame_rate=sample_rate,
                sample_width=audio.sample_width,
                channels=audio.channels,
            )

            # Save the processed audio to the output file
            processed_audio.export(output_file, format=output_file.split(".")[-1])
            print(f"Noise-reduced audio saved to {output_file}")

        except Exception as e:
            print(f"Failed to apply noise reduction to audio file: {e}")