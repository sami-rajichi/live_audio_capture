import subprocess
import re
import sys

def get_default_mic_linux():
    """Get the default microphone device on Linux using ALSA."""
    try:
        # List all audio devices
        result = subprocess.run(["arecord", "-l"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            raise RuntimeError("Failed to list audio devices using arecord.")

        # Parse the output to find the default microphone
        lines = result.stdout.splitlines()
        for line in lines:
            if "card" in line and "device" in line:
                match = re.search(r"card (\d+):.*device (\d+):", line)
                if match:
                    card, device = match.groups()
                    return f"hw:{card},{device}"
    except Exception as e:
        print(f"Error detecting default microphone: {e}")
    return "default"  # Fallback to default

def get_default_mic_mac():
    """Get the default microphone device on macOS using AVFoundation."""
    try:
        # List all audio devices
        result = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace"
        )
        if result.returncode != 0:
            raise RuntimeError("Failed to list audio devices using ffmpeg.")

        # Parse the output to find the default microphone
        lines = result.stderr.splitlines()
        for line in lines:
            if "AVFoundation audio devices" in line:
                continue
            if "[AVFoundation input device" in line and "Microphone" in line:
                match = re.search(r"\[(\d+)\]", line)
                if match:
                    return f":{match.group(1)}"  # Format for macOS
    except Exception as e:
        print(f"Error detecting default microphone: {e}")
    return ":0"  # Fallback to default

def get_default_mic_windows():
    """Get the default microphone device on Windows using DirectShow."""
    try:
        # List all audio devices
        result = subprocess.run(
            ["ffmpeg", "-f", "dshow", "-list_devices", "true", "-i", "dummy"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace"
        )
        if result.returncode != 0:
            raise RuntimeError("Failed to list audio devices using ffmpeg.")

        # Parse the output to find the default microphone
        lines = result.stderr.splitlines()
        for line in lines:
            if "DirectShow audio devices" in line:
                continue
            if "microphone" in line or "Microphone" in line or "Microphone Array" in line:
                match = re.search(r'"(.*)"', line)
                if match:
                    return f"audio={match.group(1)}"  # Format for Windows
    except Exception as e:
        print(f"Error detecting default microphone: {e}")
    return "audio=Microphone"  # Fallback to default

def get_default_mic():
    """Get the default microphone device based on the platform."""
    if sys.platform == "linux":
        return get_default_mic_linux()
    elif sys.platform == "darwin":  # macOS
        return get_default_mic_mac()
    elif sys.platform == "win32":
        return get_default_mic_windows()
    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")
