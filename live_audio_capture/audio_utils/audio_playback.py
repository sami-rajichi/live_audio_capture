# live_audio_capture/audio_utils/audio_playback.py
import numpy as np
import simpleaudio as sa
from pydub import AudioSegment

class AudioPlayback:
    """
    Utilities for playing audio files and sounds.
    """

    @staticmethod
    def play_audio_file(file_path: str) -> None:
        """
        Play an audio file using the simpleaudio library.

        Args:
            file_path (str): Path to the audio file to play.
        """
        try:
            audio = AudioSegment.from_file(file_path)
            raw_data = audio.raw_data
            play_obj = sa.play_buffer(raw_data, num_channels=audio.channels, bytes_per_sample=audio.sample_width, sample_rate=audio.frame_rate)
            play_obj.wait_done()
        except Exception as e:
            print(f"Failed to play audio file: {e}")

    @staticmethod
    def play_beep(frequency: int, duration: int) -> None:
        """
        Play a beep sound asynchronously using the simpleaudio library.

        Args:
            frequency (int): Frequency of the beep sound in Hz.
            duration (int): Duration of the beep sound in milliseconds.
        """
        try:
            sample_rate = 44100
            t = np.linspace(0, duration / 1000, int(sample_rate * duration / 1000), endpoint=False)
            waveform = np.sin(2 * np.pi * frequency * t)
            waveform = (waveform * 32767).astype(np.int16)
            play_obj = sa.play_buffer(waveform, num_channels=1, bytes_per_sample=2, sample_rate=sample_rate)
            play_obj.stop()
        except Exception as e:
            print(f"Failed to play beep sound: {e}")