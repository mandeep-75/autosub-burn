# AGENTS.md - Development Guidelines for Subtitling Tool

## Project Overview

This is a Python-based Streamlit application for automated video subtitle generation using OpenAI Whisper. It supports multilingual transcription, romanization, karaoke highlighting, and subtitle burning.

---

## 1. Build, Run, and Test Commands

### Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

### Dependencies

- `streamlit` - Web UI framework
- `faster-whisper` - GPU-accelerated Whisper transcription
- `torch` - PyTorch for GPU support
- `pysubs2` - Subtitle parsing and conversion
- `indic-transliteration` - Hindi/Punjabi romanization
- `numpy` - Numerical operations
- `requests` - HTTP requests
- `fonttools` - Font family extraction (optional)
- `watchdog` - File system monitoring (optional)

### Testing

**No formal test suite exists yet.** To add tests in the future:

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_utils.py

# Run a single test function
pytest tests/test_utils.py::test_format_timestamp
```

### Linting

No linting tool is configured. If adding one:

```bash
# Using ruff (recommended)
ruff check .
ruff format .

# Using flake8
flake8 .

# Using pylint
pylint app.py utils.py
```

### Environment Requirements

- Python 3.8+
- FFmpeg (system or in `tools/ffmpeg`)
- For GPU acceleration: CUDA-compatible GPU + torch with CUDA support

---

## 2. Code Style Guidelines

### General Principles

- Write clean, readable, and maintainable code
- Keep functions small and focused (single responsibility)
- Use descriptive names for variables, functions, and classes
- Comment complex logic but avoid obvious comments

### Import Organization

Order imports in the following groups (separate each group with a blank line):

1. Standard library imports
2. Third-party imports
3. Local application imports

```python
# Standard library
import os
import shutil
import subprocess
from typing import List, Dict, Optional, Tuple

# Third-party
import streamlit as st
import pysubs2
import torch
import faster_whisper

# Local imports
from utils import AVAILABLE_FONTS, generate_srt_content, merge_segments
```

### Type Hints

Always use type hints for function parameters and return values:

```python
# Good
def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm"""
    ...

def merge_segments(segments: List[Dict], max_words: int = 8) -> List[Dict]:
    ...

def generate_srt_content(
    segments: List[Dict],
    use_roman: bool = True,
    max_words_per_line: int = 8,
) -> str:
    ...
```

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Functions | snake_case | `merge_segments()`, `to_romanized()` |
| Variables | snake_case | `video_path`, `output_dir` |
| Constants | UPPER_SNAKE_CASE | `BASE_DIR`, `TOOLS_DIR` |
| Classes | PascalCase | (not heavily used in this project) |
| File names | snake_case | `app.py`, `utils.py` |

### Error Handling

- Use specific exception types rather than catching `Exception`
- Provide meaningful error messages
- Handle optional dependencies gracefully

```python
# Good: Handle optional dependencies gracefully
try:
    from fontTools.ttLib import TTFont
    FONTTOOLS_AVAILABLE = True
except ImportError:
    FONTTOOLS_AVAILABLE = False
    st.warning("fontTools not installed. Custom font names will be based on filenames.")

# Good: Specific exception handling
try:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
except subprocess.CalledProcessError as e:
    st.error(f"FFmpeg error: {e}")
```

### String Formatting

Use f-strings for string interpolation:

```python
# Good
output_path = os.path.join(output_dir, "final_video.mov")
srt_content += f"{count}\n{start} --> {end}\n{line}\n\n"

# Avoid
srt_content += "%d\n%s --> %s\n%s\n\n" % (count, start, end, line)
```

### Function Design

- Keep functions under 50 lines when possible
- Use default arguments for optional parameters
- Document functions with docstrings (Google style preferred)

```python
def merge_segments(segments: List[Dict], max_words: int = 8) -> List[Dict]:
    """Merge word-level segments into lines of at most max_words words.
    
    Args:
        segments: List of word-level segment dictionaries with 'start', 'end', 
                 'text', and 'words' keys.
        max_words: Maximum number of words per merged segment.
    
    Returns:
        List of merged segment dictionaries.
    """
    ...
```

### Streamlit-Specific Guidelines

- Use `st.cache_resource` for expensive model loading
- Use `st.cache_data` for expensive data transformations
- Clear caches appropriately when parameters change
- Use session state for user-specific data that persists across reruns

```python
@st.cache_resource
def get_whisper_model(model_size, device, compute_type):
    return faster_whisper.WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )
```

### Module-Level Constants

Define constants at the module level (top of file), after imports:

```python
# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(BASE_DIR, "tools")
CUSTOM_FONTS_DIR = os.path.join(BASE_DIR, "custom_fonts")
os.makedirs(CUSTOM_FONTS_DIR, exist_ok=True)
```

### Docstring Style

Use Google-style docstrings:

```python
def hex_to_ass(hex_color: str) -> str:
    """Convert #RRGGBB to ASS color format &H00BBGGRR (opaque).
    
    Args:
        hex_color: Color in hex format, with or without '#' prefix.
    
    Returns:
        ASS color format string.
    """
    hex_color = hex_color.lstrip('#')
    r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
    return f"&H00{b}{g}{r}"
```

### Avoiding Magic Numbers

Use named constants instead of magic numbers:

```python
# Good
DEFAULT_MAX_WORDS = 8
DEFAULT_FONT_SIZE = 26

# Avoid
if len(current_words) >= 8:  # What is 8?
```

### Deprecation and Version Handling

When making breaking changes, consider backward compatibility or document the change clearly.

---

## 3. Development Workflow

### Adding New Features

1. Create a new branch: `git checkout -b feature/your-feature-name`
2. Make changes following the code style guidelines
3. Test locally with `streamlit run app.py`
4. Commit with descriptive messages
5. Submit a pull request

### Debugging Tips

- Use `st.write()` or `st.text()` for quick debugging
- Check Streamlit's "Developer tools" for error traces
- Use Python's `logging` module for more complex debugging

---

## 4. File Structure

```
subtitling-tool/
├── app.py              # Main Streamlit application
├── utils.py            # Utility functions (romanization, SRT generation)
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
├── AGENTS.md           # This file
├── tools/              # External tools (ffmpeg, whisper.cpp)
├── custom_fonts/       # User-uploaded fonts
└── outputs/            # Generated subtitle files and videos
```

---

## 5. Key Implementation Details

### Whisper Model Loading

- Models are cached using `@st.cache_resource` to avoid reloading on every interaction
- Cache is cleared when model parameters change

### Subtitle Generation

- `generate_srt_content()` in `utils.py` handles all subtitle variants
- Supports plain text, karaoke highlighting, bilingual, and lip-sync modes
- Uses ASS-style inline formatting for styled subtitles

### FFmpeg Integration

- Detects system FFmpeg or uses local `tools/ffmpeg`
- Auto-selects best encoder based on system:
  - macOS: `h264_videotoolbox`
  - NVIDIA GPU: `h264_nvenc`
  - Fallback: `libx264`

---

*Last updated: March 2026*
