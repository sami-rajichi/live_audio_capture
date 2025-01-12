from live_audio_capture import LiveAudioCapture
from live_audio_capture.visualization import AudioVisualizer

# Initialize the audio capture
capture = LiveAudioCapture(sampling_rate=16000, chunk_duration=0.5, audio_format="f32le")

# Initialize the visualizer
visualizer = AudioVisualizer(sampling_rate=16000, chunk_duration=0.1)

# Stream and process audio chunks
try:
    for audio_chunk in capture.stream_audio():
        print(f"Received audio chunk with {len(audio_chunk)} samples")

        # Pass the audio chunk to the visualizer
        visualizer.add_audio_chunk(audio_chunk)
except KeyboardInterrupt:
    print("Stopping audio capture and visualizer...")
finally:
    # Stop the visualizer
    visualizer.stop()
