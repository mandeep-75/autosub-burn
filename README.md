# 🎬 Subtitling Tool

Automatically generate subtitles for your videos using **OpenAI Whisper**.

Choose between:

- 🖥 Local GPU transcription → `whisper.cpp` (macOS / Linux)
- ⚡ GPU accelerated transcription → Google Colab (`faster-whisper`) 
- Whisper.cpp can't be compiled on colab with free tier.
---


## ☁️ Google Colab (GPU Mode)

📂 **Notebook:** `colab/subtitles.ipynb`  
⚙ **Backend:** faster-whisper  
🚀 **Acceleration:** CUDA GPU  

### ▶️ Open in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mandeep-75/subtitling-tool/blob/python-whisper/colab/subtitles.ipynb)

### 🧭 Steps

1. Click the **Open in Colab** button  
2. Go to **Runtime → Change runtime type → GPU**  
3. Upload your video  
4. Run all cells  
---

## 🛠 Features

- 🎙 Automatic transcription (Whisper)
- 🌍 Multilingual support (English, Hindi, Punjabi & more)
- 🔤 Romanization (Hindi/Punjabi → Roman script)
- 🎤 Karaoke word highlighting
- 🎨 Subtitle styling (font, color, border, shadow, position)
- 🅱 Bold / Italic support
- 🎬 Burn subtitles into video
- 📏 Line-length control

---


## 🚀 Run Locally (macOS / Linux)

🍎 **Mac users → Use the `main` branch (whisper.cpp – CPU mode)**  

> ⚠️ `whisper.cpp` must be compiled from source before running the app.

### 🛠 Installation

```bash
# 1️⃣ Clone the repository
git clone https://github.com/mandeep-75/subtitling-tool.git
cd subtitling-tool

# 2️⃣ Install Python dependencies
pip install -r requirements.txt

# 3️⃣ Run the Streamlit app
streamlit run app.py
```
---

## 🤝 Contributing

Fork → Create branch → Commit → Pull request

---

## 📄 License

MIT License

---

## 📧 Contact

mandeep.dev1309@gmail.com
