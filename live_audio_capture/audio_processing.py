import numpy as np
from scipy.signal import butter, lfilter, resample
import noisereduce as nr

def apply_noise_reduction(
    audio_chunk: np.ndarray,
    sampling_rate: int,
    stationary: bool = False,
    prop_decrease: float = 1.0,
    n_std_thresh_stationary: float = 1.5,
    n_fft: int = 1024,
    win_length: int = None,
    hop_length: int = None,
    n_jobs: int = 1,  # Number of parallel jobs
    use_torch: bool = False,  # Use PyTorch for spectral gating
    device: str = "cuda",  # Device for PyTorch computation
) -> np.ndarray:
    """
    Apply noise reduction using the noisereduce package.

    Args:
        audio_chunk (np.ndarray): The audio chunk to process.
        sampling_rate (int): The sample rate of the audio.
        stationary (bool): Whether to perform stationary noise reduction.
        prop_decrease (float): Proportion to reduce noise by (1.0 = 100%).
        n_std_thresh_stationary (float): Number of standard deviations above mean for thresholding.
        n_fft (int): FFT window size.
        win_length (int): Window length for STFT.
        hop_length (int): Hop length for STFT.
        n_jobs (int): Number of parallel jobs to run. Set to -1 to use all CPU cores.
        use_torch (bool): Whether to use the PyTorch version of spectral gating.
        device (str): Device to run the PyTorch spectral gating on (e.g., "cuda" or "cpu").

    Returns:
        np.ndarray: The processed audio chunk with reduced noise.
    """
    # Apply noise reduction using noisereduce
    reduced_noise = nr.reduce_noise(
        y=audio_chunk,
        sr=sampling_rate,
        stationary=stationary,
        prop_decrease=prop_decrease,
        n_std_thresh_stationary=n_std_thresh_stationary,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_jobs=n_jobs,  # Pass the number of parallel jobs
        use_torch=use_torch,  # Enable/disable PyTorch
        device=device,  # Specify the device for PyTorch
    )
    return reduced_noise

def apply_low_pass_filter(
    audio_chunk: np.ndarray,
    sampling_rate: int,
    cutoff_freq: float = 7900.0,  # Less than Nyquist frequency (8000 Hz)
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
        raise ValueError(
            f"Cutoff frequency must be less than the Nyquist frequency ({nyquist} Hz). "
            f"Provided cutoff frequency: {cutoff_freq} Hz."
        )
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