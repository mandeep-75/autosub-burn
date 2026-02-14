import os
import subprocess
import sys
from indic_transliteration import sanscript

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(BASE_DIR, "tools")
FFMPEG_PATH = os.path.join(TOOLS_DIR, "ffmpeg")
WHISPER_ROOT = os.path.join(TOOLS_DIR, "whisper.cpp")
WHISPER_BIN = os.path.join(WHISPER_ROOT, "build", "bin", "whisper-cli")
MODELS_DIR = os.path.join(WHISPER_ROOT, "models")

# Model Metadata (Size in MB to check for corruption)
MODEL_INFO = {
    "1": ("ggml-tiny.bin", 75),
    "2": ("ggml-base.bin", 145),
    "3": ("ggml-small.bin", 480),
    "4": ("ggml-medium.bin", 1500),
    "5": ("ggml-large-v3-turbo.bin", 1600),
    "6": ("ggml-large-v3.bin", 2951)
}

def setup_whisper():
    if os.path.exists(WHISPER_BIN): return
    print("⚠️ Compiling Whisper for your hardware...")
    os.makedirs(TOOLS_DIR, exist_ok=True)
    subprocess.run(f"git clone https://github.com/ggerganov/whisper.cpp.git {WHISPER_ROOT}", shell=True)
    subprocess.run(f"cd {WHISPER_ROOT} && cmake -B build && cmake --build build --config Release", shell=True, check=True)

def download_model(choice):
    model_name, min_size = MODEL_INFO.get(choice, MODEL_INFO["2"])
    model_path = os.path.join(MODELS_DIR, model_name)
    
    # Check if exists AND if size is correct (prevents half-downloaded crashes)
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        if size_mb < (min_size - 10): # If file is too small, it's corrupted
            print(f"❌ {model_name} seems corrupted/incomplete. Redownloading...")
            os.remove(model_path)
    
    if not os.path.exists(model_path):
        print(f"📥 Downloading {model_name}...")
        script_path = os.path.join(WHISPER_ROOT, "models", "download-ggml-model.sh")
        simple_name = model_name.replace("ggml-", "").replace(".bin", "")
        subprocess.run(f"bash {script_path} {simple_name}", shell=True, cwd=WHISPER_ROOT)
    
    return model_path

def to_hinglish(text):
    roman = sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.ISO)
    mapping = {'ā': 'a', 'ī': 'i', 'ū': 'u', 'ē': 'e', 'ō': 'o', 'é': 'e', 'ḍ': 'd', 'ṛ': 'r', 'ṣ': 'sh', 'ṭ': 't', 'th': 'th'}
    for k, v in mapping.items(): roman = roman.replace(k, v)
    return roman.lower().strip()

def process_srt_to_hinglish(srt_path):
    if not os.path.exists(srt_path): return None
    with open(srt_path, "rb") as f:
        data = f.read().decode("utf-8", errors="ignore")
    lines = [to_hinglish(l) if any('\u0900' <= c <= '\u097F' for c in l) else l for l in data.splitlines()]
    with open("final_hinglish.srt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    if len(sys.argv) < 2:
        print("Usage: python srt.py <video_file>")
        return

    setup_whisper()
    
    print("\n--- Whisper Model Selection ---")
    for k, v in MODEL_INFO.items():
        print(f"{k}: {v[0]}")
    
    choice = input("\nSelect model (1-6, default 2): ").strip() or "2"
    model_path = download_model(choice)

    video_in = sys.argv[1]
    temp_wav = "temp_audio.wav"
    raw_srt = "temp_audio.wav.srt"

    # CRASH FIX: If using Large model on a laptop, disable Metal (GPU) to prevent SIGABRT
    env = os.environ.copy()
    if choice in ["4", "5", "6"]:
        print("🛡️ Large model detected. Disabling GPU (Metal) to prevent crash...")
        env["GGML_METAL_PATH_RESOURCES"] = "0" 
        env["WHISPER_METAL"] = "0"
    
    # Fix library path
    lib_path = os.path.join(WHISPER_ROOT, "build", "src")
    env["DYLD_LIBRARY_PATH"] = f"{lib_path}:{env.get('DYLD_LIBRARY_PATH', '')}"

    try:
        print("🎵 Extracting audio...")
        subprocess.run([FFMPEG_PATH, "-y", "-i", video_in, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", temp_wav], check=True)

        print("🧠 Transcribing... (Please wait)")
        # Added -ng flag to disable GPU if it's still causing issues
        cmd = [WHISPER_BIN, "-m", model_path, "-f", temp_wav, "-l", "hi", "-osrt", "-ml", "10", "-sow"]
        if choice in ["4", "5", "6"]: cmd.append("-ng") # -ng = No GPU

        subprocess.run(cmd, check=True, env=env)
        process_srt_to_hinglish(raw_srt)
        print(f"\n✅ SUCCESS! Output: final_hinglish.srt")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        for f in [temp_wav, raw_srt]:
            if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    main()