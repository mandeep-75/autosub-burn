# 🎬 Subtitling Tool

**Automatically generate subtitles for your videos using Whisper.**  
Choose between local CPU-powered transcription (via `whisper.cpp`) or GPU-accelerated processing (via Google Colab).

---

## ⚠️ Live App Notice

The [deployed Streamlit app](https://subtitling-tool-mandeep.streamlit.app/) is **for UI exploration only**.  
Due to hosting limitations:

- ❌ **Whisper transcription may not work**
- ❌ **Video burning may fail**
- ❌ **File processing may be restricted**

👉 **For full functionality, use one of the options below.**

---

## 🚀 Run Locally (macOS / Linux)

[![Mac](https://img.icons8.com/color/48/000000/mac-logo.png)](https://www.apple.com/mac/)

**🍎 Mac users** → Use the [`main`](https://github.com/your-username/subtitling-tool/tree/main) branch with `whisper.cpp` for CPU-based transcription.

### 🛠 Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/subtitling-tool.git
cd subtitling-tool

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
