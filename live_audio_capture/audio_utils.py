import numpy as np

def calculate_energy(audio_chunk: np.ndarray) -> float:
    """
    Calculate the energy of an audio chunk.

    Args:
        audio_chunk (np.ndarray): The audio chunk to process.

    Returns:
        float: The energy of the audio chunk.
    """
    return np.sum(audio_chunk**2) / len(audio_chunk)

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