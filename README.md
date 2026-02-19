# 🎬 Subtitling Tool

Automatically generate subtitles for your videos using Whisper. Choose between local CPU-powered transcription or GPU-accelerated processing (via Google Colab).


## 👉 For full functionality, use one of the options below.

## 🚀 Run Locally (macOS / Linux)

[![macOS Logo](https://img.icons8.com/color/48/000000/mac-logo.png)](https://www.apple.com/macos/)

### 🍎 Mac users → Use the main branch with whisper.cpp for GPU-based transcription.

#### 🛠 Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/subtitling-tool.git
cd subtitling-tool

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py