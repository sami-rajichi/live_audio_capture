# examples/realtime_visualization.py
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
