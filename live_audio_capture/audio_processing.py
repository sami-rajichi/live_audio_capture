import numpy as np
from scipy.signal import butter, lfilter, resample

def apply_noise_reduction(audio_chunk: np.ndarray, sampling_rate: int) -> np.ndarray:
    """
    Apply a basic noise reduction filter to the audio chunk.

    Args:
        audio_chunk (np.ndarray): The audio chunk to process.
        sampling_rate (int): The sample rate of the audio.

    Returns:
        np.ndarray: The processed audio chunk.
    """
    # Design a Butterworth low-pass filter
    nyquist = 0.5 * sampling_rate
    normal_cutoff = 4000 / nyquist  # Cutoff frequency of 4 kHz
    b, a = butter(5, normal_cutoff, btype="low", analog=False)
    return lfilter(b, a, audio_chunk)

def resample_audio(audio_chunk: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    """
    Resample the audio chunk to a target sample rate.

    Args:
        audio_chunk (np.ndarray): The audio chunk to resample.
        original_rate (int): The original sample rate.
        target_rate (int): The target sample rate.

    Returns:
        np.ndarray: The resampled audio chunk.
    """
    num_samples = int(len(audio_chunk) * target_rate / original_rate)
    return resample(audio_chunk, num_samples)