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





#------------------------------------------------

from live_audio_capture import LiveAudioCapture

# Initialize the audio capture
capture = LiveAudioCapture(
    sampling_rate=16000,
    chunk_duration=0.3,
    audio_format="f32le",
    channels=1,
    aggressiveness=1,  
    enable_beep=True,
    low_pass_cutoff=7500,
    stationary_noise_reduction=False,
    enable_noise_canceling=False,
    calibration_duration=4.0,
    use_adaptive_threshold=True,
)

# List available microphones
# mics = capture.list_available_mics()
# print("Available microphones:", mics)

# # Change input device by name
# capture.change_input_device("Réseau de microphones (Realtek(R) Audio)")  # Replace with a valid microphone name

# Record audio with advanced VAD and save to a file
capture.listen_and_record_with_vad(
    output_file="transcription.wav", 
    silence_duration=2, 
    format='wav'
)

# # Play an audio file
# capture.play_audio_file("transcription.wav")