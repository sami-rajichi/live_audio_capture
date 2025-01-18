# Usage

## Basic Example

Capture audio with voice activity detection and save it to a file:

```python
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

```

---

## Real-Time Visualization

Visualize audio in real-time:

```python
import time
from live_audio_capture import LiveAudioCapture, AudioVisualizer
import threading

# Initialize the audio capture
capture = LiveAudioCapture(
    sampling_rate=44100,  # Higher sample rate for better visualization
    chunk_duration=0.1,
    audio_format="f32le",
    channels=1,
    enable_noise_canceling=False,  # Disable noise cancellation
)

# Initialize the audio visualizer
visualizer = AudioVisualizer(
    sampling_rate=44100,
    chunk_duration=0.1,
)


# Function to stream audio and visualize it
def stream_and_visualize():
    for audio_chunk in capture.stream_audio():
        visualizer.add_audio_chunk(audio_chunk)


# Start the visualization in a separate thread
visualizer_thread = threading.Thread(target=stream_and_visualize)
visualizer_thread.start()

# Let the visualization run for 10 seconds
time.sleep(10)

# Stop the capture and visualization
capture.stop()
visualizer.stop()
```

---

## Advanced Example

Use the real-time audio capture along with the voice activity detector within a thread (It's highly recommended to follow this approach):

```python
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
```

---

## Noise Reduction Example

Apply the noise reducer feature on a noisy-environment recorded sample:

```python
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
```

---

### Change Input Device

Modify the microphone name if needed:

```python
from live_audio_capture import LiveAudioCapture

# Initialize the audio capture
capture = LiveAudioCapture(
    sampling_rate=16000,
    chunk_duration=0.1,
    audio_format="f32le",
    channels=1,
)

# List available microphones
mics = capture.list_available_mics()
print("Available microphones:")
for mic_name, device_id in mics.items():
    print(f"{mic_name}: {device_id}")

# Change the input device to the first available microphone
if mics:
    first_mic_name = list(mics.keys())[0]
    capture.change_input_device(first_mic_name)
    print(f"Changed input device to: {first_mic_name}")

# Start recording with VAD
capture.listen_and_record_with_vad(
    output_file="output.wav",
    silence_duration=2.0,
    format="wav",
)
```
