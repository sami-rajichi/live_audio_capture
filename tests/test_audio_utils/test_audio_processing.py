# tests/test_audio_processing.py
import unittest
import os
import numpy as np
from live_audio_capture.audio_utils.audio_processing import AudioProcessing

class TestAudioProcessing(unittest.TestCase):
    def test_calculate_energy(self):
        """Test calculating the energy of an audio chunk."""
        audio_chunk = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        energy = AudioProcessing.calculate_energy(audio_chunk)
        self.assertAlmostEqual(energy, 0.075, places=3)

    def test_process_audio_chunk(self):
        """Test processing raw audio data into a NumPy array."""
        raw_data = b"\x00\x00\x00\x00\x00\x00\x80\x3F"  # 1.0 in float32
        audio_chunk = AudioProcessing.process_audio_chunk(raw_data, "f32le")
        self.assertEqual(audio_chunk.dtype, np.float32)
        self.assertAlmostEqual(audio_chunk[1], 1.0)

    def test_apply_noise_reduction_to_file(self):
        """Test applying noise reduction to an audio file."""
        # Create a dummy input file (you can replace this with a real file)
        input_file = "./tests/test_audio_utils/test_input.wav"
        output_file = "./tests/test_audio_utils/test_output.wav"
        
        # Apply noise reduction
        AudioProcessing.apply_noise_reduction_to_file(
            input_file=input_file,
            output_file=output_file,
            stationary=True,
            prop_decrease=1.0,
            n_std_thresh_stationary=1.5,
            n_jobs=1,
            use_torch=False,
            device="cpu",
        )

        # Check if the output file was created
        self.assertTrue(os.path.exists(output_file))

        # Clean up output file
        os.remove(output_file)

if __name__ == "__main__":
    unittest.main()