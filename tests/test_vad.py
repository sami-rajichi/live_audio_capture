# tests/test_vad.py
import unittest
import numpy as np
from live_audio_capture import VoiceActivityDetector


class TestVoiceActivityDetector(unittest.TestCase):
    def test_initialization(self):
        """Test initializing the VoiceActivityDetector class."""
        vad = VoiceActivityDetector(
            sample_rate=16000,
            frame_duration=0.03,
            aggressiveness=1,
        )
        self.assertIsNotNone(vad)

    def test_process_audio(self):
        """Test processing an audio chunk with VAD."""
        vad = VoiceActivityDetector(
            sample_rate=16000,
            frame_duration=0.03,
            aggressiveness=1,
        )
        audio_chunk = np.random.randn(480).astype(np.float32)  # Simulate 30ms of audio
        is_speech = vad.process_audio(audio_chunk)
        self.assertIsInstance(is_speech, bool)


if __name__ == "__main__":
    unittest.main()
