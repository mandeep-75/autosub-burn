## 🎬 Subtitling Tool
Automatically generate subtitles for your videos using Whisper.
Choose between local CPU-powered transcription (via whisper.cpp) or GPU-accelerated processing (via Google Colab).

## ⚠️ Live App Notice
The deployed Streamlit app is for UI exploration only.
Due to hosting limitations:

❌ Whisper transcription may not work

❌ Video burning may fail

❌ File processing may be restricted

## 👉 For full functionality, use one of the options below.

## 🚀 Run Locally (macOS / Linux)
https://img.icons8.com/color/48/000000/mac-logo.png

## 🍎 Mac users → Use the main branch with whisper.cpp for CPU-based transcription.

🛠 Installation Steps
bash
# 1. Clone the repository
git clone https://github.com/your-username/subtitling-tool.git
cd subtitling-tool

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
## ☁️ Google Colab (GPU Mode)
For faster processing on long videos, use the GPU-accelerated notebook:

Notebook: colab/subtitles.ipynb

Backend: faster-whisper (Python package)

Acceleration: GPU (CUDA)

Steps
Open the notebook in Google Colab

Enable GPU runtime (Runtime → Change runtime type → GPU)

Upload your video

Run all cells to generate subtitles

## 🛠️ Features
Automatic Transcription: Whisper-based subtitle generation

Language Support: Multilingual transcription (English, Hindi, Punjabi, etc.)

Romanization: Transliterates Hindi/Punjabi to Roman script

Karaoke Highlighting: Highlights words as they're spoken

Customization: Control line length, fonts, colors, borders, shadows, and positioning

Font Options: Multiple fonts with bold/italic support

Video Burning: Burns subtitles directly into the video

## 📂 Project Structure
```bash
subtitling-tool/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── colab/                  # Google Colab notebooks
│   └── subtitles.ipynb     # GPU-accelerated version
├── tools/                  # FFmpeg binary (macOS)
├── outputs/                # Generated videos and subtitles
└── README.md               # Project documentation
```
## 🤝 Contributing
Contributions are welcome! Feel free to fork the repository, create a feature branch, and submit a pull request.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact
For questions or support, please open an issue or contact mandeep.dev1309@gmail.com.