# Contributing to Live Audio Capture

Thank you for your interest in contributing to **Live Audio Capture**! We welcome contributions from everyone, whether you're fixing a bug, adding a feature, or improving documentation. This guide will help you get started.

---

## Table of Contents
1. [Getting Started](#getting-started)
2. [Setting Up the Development Environment](#setting-up-the-development-environment)
3. [Making Changes](#making-changes)
4. [Coding Standards](#coding-standards)
5. [Testing](#testing)
6. [Submitting a Pull Request](#submitting-a-pull-request)
7. [Reporting Issues](#reporting-issues)

---

## Getting Started

Before you start contributing, please:
1. **Fork the repository** on GitHub.
2. **Clone your fork** to your local machine:
   ```bash
   git clone https://github.com/sami-rajichi/live_audio_capture.git
   ```
3. **Create a new branch** for your changes:
   ```bash
   git checkout -b your-branch-name
   ```

---

## Setting Up the Development Environment

1. **Install Python**: Ensure you have Python 3.9 or higher installed.
2. **Install Dependencies**:
   - Navigate to the project directory:
     ```bash
     cd live_audio_capture
     ```
   - Install the development dependencies:
     ```bash
     pip install -r requirements.txt
     pip install -e .
     ```
3. **Install FFmpeg**: Follow the installation instructions in the [README.md](README.md).

---

## Making Changes

1. **Follow the Project Structure**:
   - Code lives in the `live_audio_capture/` directory.
   - Tests are in the `tests/` directory.
   - Examples are in the `examples/` directory.
2. **Write Clear and Concise Code**: Ensure your code is easy to read and understand.
3. **Add Documentation**: Update the relevant documentation (e.g., `README.md`, docstrings) if your changes introduce new features or modify existing behavior.

---

## Coding Standards

1. **Style**: Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code.
2. **Type Annotations**: Use type hints for function arguments and return values.
3. **Docstrings**: Include docstrings for all public functions, classes, and methods. Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstring format.
4. **Commit Messages**: Write clear and descriptive commit messages. Use the present tense (e.g., "Add feature X" instead of "Added feature X").

---

## Testing

1. **Write Tests**: Ensure your changes are covered by unit tests. Place new test files in the `tests/` directory.
2. **Run Tests**: Use `pytest` to run the test suite:
   ```bash
   pytest tests/
   ```
3. **Check Coverage**: Ensure your changes do not reduce test coverage:
   ```bash
   pytest --cov=live_audio_capture tests/
   ```

---

## Submitting a Pull Request

1. **Push Your Changes**: Push your branch to your fork on GitHub:
   ```bash
   git push origin your-branch-name
   ```
2. **Open a Pull Request**:
   - Go to the [GitHub repository](https://github.com/sami-rajichi/live_audio_capture).
   - Click **New Pull Request**.
   - Select your branch and provide a clear description of your changes.
3. **Address Feedback**: Be responsive to feedback and make necessary updates to your pull request.

---

## Reporting Issues

If you encounter a bug or have a feature request, please:
1. **Search Existing Issues**: Check if the issue has already been reported.
2. **Open a New Issue**: Provide a clear and detailed description of the problem or feature request.
   - For bugs, include steps to reproduce the issue.
   - For features, explain the use case and why it would be valuable.

---

## Thank You!

Your contributions help make **Live Audio Capture** better for everyone. We appreciate your time and effort!
