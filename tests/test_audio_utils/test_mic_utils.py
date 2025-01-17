# tests/test_mic_utils.py
import unittest
from live_audio_capture.audio_utils.mic_utils import MicUtils


class TestMicUtils(unittest.TestCase):
    def test_list_mics(self):
        """Test listing available microphones."""
        mics = MicUtils.list_mics()
        self.assertIsInstance(mics, dict)
        if mics:  # If microphones are available
            self.assertTrue(all(isinstance(key, str) for key in mics.keys()))
            self.assertTrue(all(isinstance(value, str) for value in mics.values()))

    def test_get_default_mic(self):
        """Test getting the default microphone."""
        default_mic = MicUtils.get_default_mic()
        self.assertIsInstance(default_mic, str)


if __name__ == "__main__":
    unittest.main()
