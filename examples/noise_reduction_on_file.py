# examples/noise_reduction_on_file.py
from live_audio_capture.audio_utils import AudioProcessing, AudioPlayback

# Apply noise reduction to a pre-recorded file
AudioProcessing.apply_noise_reduction_to_file(
    input_file="input.wav",  # Path to the input file
    output_file="output.wav",  # Path to save the processed file
    stationary=True,  # Enable stationary noise reduction
    prop_decrease=1.0,  # Reduce noise by 100%
    n_std_thresh_stationary=1.5,  # Threshold for stationary noise reduction
    n_jobs=1,  # Use a single job for noise reduction
    use_torch=False,  # Disable PyTorch for noise reduction
    device="cpu",  # Use CPU for noise reduction
)

AudioPlayback.play_audio_file("output.wav")

print("Noise reduction applied and saved to output.wav.")