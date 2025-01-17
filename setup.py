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
    url="https://github.com/sami-rajichi/live_audio_capture",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.4",
        "scipy>=1.12.0",
        "pydub>=0.25.1",
        "pyqtgraph>=0.13.7",
        "noisereduce>=3.0.3",
        "sounddevice>=0.4.6",
        "simpleaudio>=1.0.4"
    ],
    extras_require={
        "dev": [
            "pytest>=8.3.4",
            "flake8>=7.1.1",
            "twine>=6.0.1",
        ],
        "torch": [
            "torch>=2.2.1",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="audio capture ffmpeg real-time visualization",
    test_suite="tests",
    tests_require=["pytest>=8.3.0"],
    python_requires=">=3.9",
    project_urls={
        "Bug Reports": "https://github.com/sami-rajichi/live_audio_capture/issues",
        "Source": "https://github.com/sami-rajichi/live_audio_capture"
    }
)
