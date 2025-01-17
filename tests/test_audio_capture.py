# tests/test_audio_capture.py
import unittest
from live_audio_capture import LiveAudioCapture


class TestLiveAudioCapture(unittest.TestCase):
    def test_initialization(self):
        """Test initializing the LiveAudioCapture class."""
        capture = LiveAudioCapture(
            sampling_rate=16000,
            chunk_duration=0.1,
            audio_format="f32le",
            channels=1,
        )
        self.assertIsNotNone(capture)

    def test_list_available_mics(self):
        """Test listing available microphones."""
        capture = LiveAudioCapture()
        mics = capture.list_available_mics()
        self.assertIsInstance(mics, dict)

    def test_change_input_device(self):
        """Test changing the input device."""
        capture = LiveAudioCapture()
        mics = capture.list_available_mics()
        if mics:  # If microphones are available
            first_mic_name = list(mics.keys())[0]
            capture.change_input_device(first_mic_name)
            self.assertEqual(capture.input_device, mics[first_mic_name])


if __name__ == "__main__":
    unittest.main()
