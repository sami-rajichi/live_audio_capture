import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from threading import Thread
import queue
from scipy.signal import spectrogram
from typing import Optional


class AudioVisualizer:
    """
    A standalone real-time audio visualizer using PyQtGraph with a refined color scheme.
    """

    def __init__(self, sampling_rate: int, chunk_duration: float):
        """
        Initialize the AudioVisualizer instance.

        Args:
            sampling_rate (int): The sample rate of the audio.
            chunk_duration (float): The duration of each audio chunk in seconds.
        """
        self.sampling_rate = sampling_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sampling_rate * chunk_duration)

        # Create a thread-safe queue for audio chunks
        self.audio_queue = queue.Queue()

        # Flag to control the visualization thread
        self.running = False

        # Start the visualization thread
        self.thread = Thread(target=self._run_visualizer, daemon=True)
        self.thread.start()

    def _run_visualizer(self):
        """Run the PyQtGraph visualizer in a separate thread."""
        # Create a PyQtGraph application
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title="Real-Time Audio Visualizer")
        self.win.resize(1200, 800)  # Increase window size for additional plots
        self.win.setBackground("#1f1f1f")  # Dark gray background
        self.win.show()

        # Custom font for labels
        _ = QtGui.QFont("Arial", 12)
        pg.setConfigOptions(antialias=True, useNumba=True)

        # Create a plot for the waveform
        self.waveform_plot = self.win.addPlot(title="Waveform", row=0, col=0)
        self.waveform_curve = self.waveform_plot.plot(pen=pg.mkPen("#00ffff", width=2))  # Soft cyan
        self.waveform_plot.setYRange(-1, 1)
        self.waveform_plot.setXRange(0, self.chunk_size)
        self.waveform_plot.setTitle("Waveform", color="#ffffff", size="14pt")
        self.waveform_plot.setLabel("left", "Amplitude", color="#ffffff", **{"font-size": "12pt"})
        self.waveform_plot.setLabel("bottom", "Time (samples)", color="#ffffff", **{"font-size": "12pt"})

        # Create a plot for the frequency spectrum
        self.spectrum_plot = self.win.addPlot(title="Frequency Spectrum", row=0, col=1)
        self.spectrum_curve = self.spectrum_plot.plot(pen=pg.mkPen("#ff00ff", width=2))  # Soft magenta
        self.spectrum_plot.setLogMode(x=True, y=False)  # Logarithmic frequency axis
        self.spectrum_plot.setLabel("left", "Magnitude (dB)", color="#ffffff", **{"font-size": "12pt"})
        self.spectrum_plot.setLabel("bottom", "Frequency (Hz)", color="#ffffff", **{"font-size": "12pt"})
        self.spectrum_plot.setYRange(-100, 0)
        self.spectrum_plot.setXRange(20, self.sampling_rate / 2)  # 20 Hz to Nyquist frequency
        self.spectrum_plot.setTitle("Frequency Spectrum", color="#ffffff", size="14pt")

        # Add a peak frequency indicator
        self.peak_freq_text = pg.TextItem(anchor=(0.5, 1), color="#ff00ff")  # Soft magenta
        self.spectrum_plot.addItem(self.peak_freq_text)

        # Create a plot for the spectrogram
        self.spectrogram_plot = self.win.addPlot(title="Spectrogram", row=1, col=0, colspan=2)
        self.spectrogram_image = pg.ImageItem()
        self.spectrogram_plot.addItem(self.spectrogram_image)
        self.spectrogram_plot.setLabel("left", "Frequency (Hz)", color="#ffffff", **{"font-size": "12pt"})
        self.spectrogram_plot.setLabel("bottom", "Time (s)", color="#ffffff", **{"font-size": "12pt"})
        self.spectrogram_plot.setTitle("Spectrogram", color="#ffffff", size="14pt")

        # Set a color map for the spectrogram (deep blue to bright yellow)
        self.colormap = pg.ColorMap(
            [0.0, 1.0],  # Positions for the colors
            [
                (0, 0, 255),  # Deep blue at position 0.0
                (255, 255, 0),  # Bright yellow at position 1.0
            ]
        )
        self.spectrogram_image.setLookupTable(self.colormap.getLookupTable())

        # Initialize spectrogram data
        self.spectrogram_data = np.zeros((129, 100))  # 129 frequency bins, 100 time steps
        self.spectrogram_image.setImage(self.spectrogram_data)
        self.spectrogram_image.setLevels([-50, 0])  # Adjust levels for better contrast

        # Create a volume meter
        self.volume_meter = self.win.addPlot(title="Volume Meter", row=2, col=0)
        self.volume_bar = pg.BarGraphItem(x=[0], height=[0], width=0.6, brush="#ff8c42")  # Gradient orange
        self.volume_meter.addItem(self.volume_bar)
        self.volume_meter.setYRange(0, 1)
        self.volume_meter.setXRange(-1, 1)
        self.volume_meter.setTitle("Volume Meter", color="#ffffff", size="14pt")
        self.volume_meter.setLabel("left", "Volume", color="#ffffff", **{"font-size": "12pt"})

        # Create a volume history plot
        self.volume_history_plot = self.win.addPlot(title="Volume History", row=2, col=1)
        self.volume_history_curve = self.volume_history_plot.plot(pen=pg.mkPen("#00ff00", width=2))  # Soft green
        self.volume_history_plot.setYRange(0, 1)
        self.volume_history_plot.setXRange(0, 100)  # Show last 100 volume readings
        self.volume_history_plot.setTitle("Volume History", color="#ffffff", size="14pt")
        self.volume_history_plot.setLabel("left", "Volume", color="#ffffff", **{"font-size": "12pt"})
        self.volume_history_plot.setLabel("bottom", "Time (samples)", color="#ffffff", **{"font-size": "12pt"})

        # Timer for updating the visualization
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(100)  # Update every 100 ms

        # Start the Qt event loop
        self.running = True
        self.app.exec()

    def _update(self):
        """Update the visualization with the latest audio chunk."""
        try:
            # Get the latest audio chunk from the queue
            audio_chunk = self.audio_queue.get_nowait()

            # Update waveform
            self.waveform_curve.setData(audio_chunk)

            # Update frequency spectrum
            spectrum = self.compute_spectrum(audio_chunk)
            if spectrum is not None:
                freqs = np.fft.rfftfreq(len(audio_chunk), 1 / self.sampling_rate)
                self.spectrum_curve.setData(freqs, spectrum)

                # Update peak frequency indicator
                peak_freq = freqs[np.argmax(spectrum)]
                self.peak_freq_text.setText(f"Peak: {peak_freq:.1f} Hz")
                self.peak_freq_text.setPos(peak_freq, np.max(spectrum))

            # Update spectrogram
            spectrogram_chunk = self.compute_spectrogram(audio_chunk)
            if spectrogram_chunk is not None:
                self.spectrogram_data = np.roll(self.spectrogram_data, -1, axis=1)
                self.spectrogram_data[:, -1] = spectrogram_chunk
                self.spectrogram_image.setImage(self.spectrogram_data, autoLevels=False)

            # Update volume meter
            volume = np.sqrt(np.mean(audio_chunk**2))  # RMS volume
            self.volume_bar.setOpts(height=[volume])

            # Update volume history
            if not hasattr(self, "volume_history"):
                self.volume_history = np.zeros(100)
            self.volume_history = np.roll(self.volume_history, -1)
            self.volume_history[-1] = volume
            self.volume_history_curve.setData(self.volume_history)

        except queue.Empty:
            # No new audio chunk available
            pass

    def compute_spectrum(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute the frequency spectrum for a given audio chunk.

        Args:
            audio_chunk (np.ndarray): The audio chunk to process.

        Returns:
            Optional[np.ndarray]: The frequency spectrum in dB.
        """
        try:
            # Compute the FFT
            fft = np.fft.rfft(audio_chunk)
            magnitude = np.abs(fft)
            return 10 * np.log10(magnitude + 1e-10)  # Convert to dB
        except Exception:
            return None

    def compute_spectrogram(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute the spectrogram for a given audio chunk.

        Args:
            audio_chunk (np.ndarray): The audio chunk to process.

        Returns:
            Optional[np.ndarray]: The spectrogram data.
        """
        try:
            _, _, Sxx = spectrogram(audio_chunk, fs=self.sampling_rate, nperseg=256)
            return 10 * np.log10(Sxx.mean(axis=1) + 1e-10)  # Convert to dB
        except ValueError:
            # Handle cases where the audio chunk is too short for the spectrogram
            return None

    def add_audio_chunk(self, audio_chunk: np.ndarray):
        """Add a new audio chunk to the visualization queue."""
        self.audio_queue.put(audio_chunk)

    def stop(self):
        """Stop the visualization."""
        self.running = False
        self.app.quit()
