from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
setup(
    name="live_audio_capture",
    version="0.4.1",
    author="Sami RAJICHI",
    author_email="semi.rajichi@gmail.com",
    description="A cross-platform utility for capturing live audio from a microphone using FFmpeg.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sami-rajichi/live_audio_capture",  # Link to your project's repository
    license="MIT",  # License type
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "pydub",
        "pyqtgraph",
        "noisereduce",
        "sounddevice",
        "simpleaudio"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",  # Project maturity
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="audio capture ffmpeg real-time visualization",
    test_suite="tests",
    tests_require=["pytest>=6.0.0"],
)
