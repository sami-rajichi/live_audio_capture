# examples/threaded_vad_recording.py
from live_audio_capture import LiveAudioCapture
import threading
import time

# Initialize the audio capture
capture = LiveAudioCapture(
    sampling_rate=16000,
    chunk_duration=0.3,
    audio_format="f32le",
    channels=1,
    aggressiveness=1,
    enable_beep=True,
    enable_noise_canceling=False,
    low_pass_cutoff=7500.0,
    stationary_noise_reduction=True,
    prop_decrease=1.0,
    n_std_thresh_stationary=1.5,
    n_jobs=1,
    use_torch=False,
    device="cpu",
    calibration_duration=2.0,
    use_adaptive_threshold=True,
)


# Function to start recording with VAD
def record_with_vad():
    capture.listen_and_record_with_vad(
        output_file="output.wav",
        silence_duration=2.0,
        format="wav"
    )


# Start the recording in a separate thread
recording_thread = threading.Thread(target=record_with_vad)
recording_thread.start()

# Let the recording run for 10 seconds
time.sleep(10)

# Stop the recording
capture.stop()

print("Recording stopped.")
