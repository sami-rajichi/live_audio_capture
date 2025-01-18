# examples/change_input_device.py
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
