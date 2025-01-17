# tests/test_audio_playback.py
import os
import unittest
from live_audio_capture.audio_utils.audio_playback import AudioPlayback


class TestAudioPlayback(unittest.TestCase):
    def test_play_audio_file(self):
        """Test playing an audio file."""
        # Create a dummy audio file (you can replace this with a real file)
        input_file = "test_audio.wav"
        with open(input_file, "wb") as f:
            f.write(b"Dummy audio data")  # Replace with actual audio data

        # Play the audio file (this test only checks if the method runs without errors)
        AudioPlayback.play_audio_file(input_file)

        # Clean up test file
        os.remove(input_file)

    def test_play_beep(self):
        """Test playing a beep sound."""
        # Play a beep (this test only checks if the method runs without errors)
        AudioPlayback.play_beep(frequency=440, duration=100)


if __name__ == "__main__":
    unittest.main()
