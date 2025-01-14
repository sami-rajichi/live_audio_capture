# from live_audio_capture import LiveAudioCapture
# from live_audio_capture.visualization import AudioVisualizer

# # Initialize the audio capture
# capture = LiveAudioCapture(sampling_rate=16000, chunk_duration=0.5, audio_format="f32le")

# # Initialize the visualizer
# visualizer = AudioVisualizer(sampling_rate=16000, chunk_duration=0.1)

# # Stream and process audio chunks
# try:
#     for audio_chunk in capture.stream_audio():
#         print(f"Received audio chunk with {len(audio_chunk)} samples")

#         # Pass the audio chunk to the visualizer
#         visualizer.add_audio_chunk(audio_chunk)
# except KeyboardInterrupt:
#     print("Stopping audio capture and visualizer...")
# finally:
#     # Stop the visualizer
#     visualizer.stop()








# from live_audio_capture import LiveAudioCapture

# # Initialize the audio capture
# capture = LiveAudioCapture(
#     sampling_rate=16000,
#     chunk_duration=1,
#     vad_threshold=0.001,
#     noise_floor_alpha=0.9,
#     hysteresis_high=1.5,
#     hysteresis_low=0.5,
#     enable_beep=True
# )

# # Record audio with advanced VAD and save to a file
# capture.listen_and_record_with_vad(output_file="transcription.mp3", silence_duration=2, format='mp3')







from live_audio_capture import LiveAudioCapture

# Initialize the audio capture
capture = LiveAudioCapture(
    sampling_rate=16000,
    chunk_duration=1,
    vad_threshold=0.0005,
    noise_floor_alpha=0.9,
    hysteresis_high=1.5,
    hysteresis_low=0.5,
    enable_beep=True,
    noise_threshold_db= -50.0,
    low_pass_cutoff=  7800.0
)

# Record audio with advanced VAD and save to a file
capture.listen_and_record_with_vad(
    output_file="transcription.wav", 
    silence_duration=2, 
    format='wav', 
    enable_noise_canceling=True
)
