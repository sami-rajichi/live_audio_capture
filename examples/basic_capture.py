# examples/basic_capture.py
from live_audio_capture import LiveAudioCapture

# Initialize the audio capture
capture = LiveAudioCapture(
    sampling_rate=16000,  # Sample rate in Hz
    chunk_duration=0.3,   # Duration of each audio chunk in seconds
    audio_format="f32le",  # Audio format
    channels=1,           # Mono audio
    aggressiveness=1,     # VAD aggressiveness level
    enable_beep=True,     # Play beep sounds when recording starts/stops
    enable_noise_canceling=True,  # Enable noise cancellation
    low_pass_cutoff=7500.0,       # Low-pass filter cutoff frequency
    stationary_noise_reduction=True,  # Enable stationary noise reduction
    prop_decrease=1.0,    # Proportion to reduce noise by
    n_std_thresh_stationary=1.5,  # Threshold for stationary noise reduction
    n_jobs=1,             # Number of parallel jobs
    use_torch=False,      # Disable PyTorch for noise reduction
    device="cpu",         # Use CPU for noise reduction
    calibration_duration=2.0,  # Calibration duration in seconds
    use_adaptive_threshold=True,  # Enable adaptive thresholding for VAD
)

# Start recording with VAD
capture.listen_and_record_with_vad(
    output_file="output.wav",  # Save the recording to this file
    silence_duration=2.0,      # Stop recording after 2 seconds of silence
    format="wav",              # Output format
)
