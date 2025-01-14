import numpy as np
from scipy.signal import butter, lfilter, stft, istft, resample
from typing import Optional, Tuple

def apply_noise_reduction(
    audio_chunk: np.ndarray,
    sampling_rate: int,
    noise_threshold_db: float = -20.0,  # Less aggressive threshold
    nperseg: int = 512,  # Larger window size for better frequency resolution
) -> np.ndarray:
    """
    Apply spectral gating noise reduction to the audio chunk.

    Args:
        audio_chunk (np.ndarray): The audio chunk to process.
        sampling_rate (int): The sample rate of the audio.
        noise_threshold_db (float): The noise threshold in dB. Frequencies below this threshold will be attenuated.
        nperseg (int): STFT window size.

    Returns:
        np.ndarray: The processed audio chunk with reduced noise.
    """
    # Compute the Short-Time Fourier Transform (STFT)
    f, t, Zxx = stft(audio_chunk, fs=sampling_rate, nperseg=nperseg, noverlap=nperseg // 2)

    # Convert the STFT magnitude to dB
    magnitude = np.abs(Zxx)
    magnitude_db = 20 * np.log10(magnitude + 1e-10)  # Add small epsilon to avoid log(0)

    # Create a noise mask based on the threshold
    noise_mask = magnitude_db > noise_threshold_db

    # Apply the noise mask to the STFT
    Zxx_denoised = Zxx * noise_mask

    # Compute the Inverse Short-Time Fourier Transform (ISTFT) to reconstruct the audio
    _, audio_denoised = istft(Zxx_denoised, fs=sampling_rate, nperseg=nperseg, noverlap=nperseg // 2)

    return audio_denoised

def apply_low_pass_filter(
    audio_chunk: np.ndarray,
    sampling_rate: int,
    cutoff_freq: float = 8000.0,  # Higher cutoff frequency for better voice clarity
) -> np.ndarray:
    """
    Apply a low-pass filter to the audio chunk.

    Args:
        audio_chunk (np.ndarray): The audio chunk to process.
        sampling_rate (int): The sample rate of the audio.
        cutoff_freq (float): The cutoff frequency for the low-pass filter.

    Returns:
        np.ndarray: The filtered audio chunk.
    """
    # Normalize the cutoff frequency to the range [0, 1]
    nyquist = 0.5 * sampling_rate
    if cutoff_freq >= nyquist:
        raise ValueError(f"Cutoff frequency must be less than the Nyquist frequency ({nyquist} Hz).")
    normal_cutoff = cutoff_freq / nyquist

    # Design the Butterworth filter
    b, a = butter(5, normal_cutoff, btype="low", analog=False)

    # Apply the filter to the audio chunk
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