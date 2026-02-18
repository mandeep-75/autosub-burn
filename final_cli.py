import argparse
import sys
import os
import shutil
import subprocess
import json
from utils import generate_srt_content, AVAILABLE_FONTS, parse_whisper_json, parse_srt_to_segments, merge_segments, get_best_ffmpeg_encoder, get_whisper_gpu_args

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(BASE_DIR, "tools")
WHISPER_CPP_DIR = os.path.join(TOOLS_DIR, "whisper.cpp")
WHISPER_CLI = os.path.join(WHISPER_CPP_DIR, "build/bin/whisper-cli")
MODELS_DIR = os.path.join(WHISPER_CPP_DIR, "models")

# Robust FFmpeg finding
SYSTEM_FFMPEG = shutil.which("ffmpeg")
LOCAL_FFMPEG = os.path.join(TOOLS_DIR, "ffmpeg")

# Prefer local ffmpeg if it exists, then system
if os.path.exists(LOCAL_FFMPEG):
    FFMPEG_PATH = LOCAL_FFMPEG
elif SYSTEM_FFMPEG:
    FFMPEG_PATH = SYSTEM_FFMPEG
else:
    FFMPEG_PATH = "ffmpeg"

WHISPER_MODELS = {
    "tiny": "ggml-tiny.bin",
    "base": "ggml-base.bin",
    "small": "ggml-small.bin",
    "medium": "ggml-medium.bin",
    "large-v3": "ggml-large-v3.bin",
    "large-v3-turbo": "ggml-large-v3-turbo.bin"
}

def ensure_model_exists(model_name):
    filename = WHISPER_MODELS.get(model_name)
    if not filename:
        return None
    model_path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(model_path):
        return model_path
    
    print(f"📥 Downloading {model_name} model...")
    url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{filename}"
    try:
        import requests
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return model_path
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="AI Subtitle Generator CLI (Word-Level Lip Sync)")
    
    # Required
    parser.add_argument("video_path", help="Path to input video file")
    
    # Optional
    parser.add_argument("--model", default="base", choices=list(WHISPER_MODELS.keys()), help="Whisper model size (default: base/multilingual)")
    parser.add_argument("--roman", action="store_true", default=True, help="Transliterate to Roman")
    parser.add_argument("--max-words", type=int, default=8, help="Max words per line")
    parser.add_argument("--font-size", type=int, default=26, help="Font size")
    parser.add_argument("--base-font", default="Arial", choices=list(AVAILABLE_FONTS.keys()), help="Base font")
    parser.add_argument("--highlight-color", default="#FFFF00", help="Highlight color")

    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"❌ Error: Video not found at {args.video_path}")
        return

    # Create Output Directory
    video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
    output_dir = os.path.join(BASE_DIR, "outputs", video_basename)
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy original video to output folder
    original_video_in_output = os.path.join(output_dir, "original" + os.path.splitext(args.video_path)[1])
    shutil.copy2(args.video_path, original_video_in_output)

    if not os.path.exists(WHISPER_CLI):
        print(f"❌ Error: whisper-cli not found at {WHISPER_CLI}")
        return

    model_path = ensure_model_exists(args.model)
    if not model_path: return

    print(f"🎬 Processing: {args.video_path}")
    
    # 1. Extract Audio
    temp_audio = os.path.join(output_dir, "audio.wav")
    subprocess.run([FFMPEG_PATH, "-y", "-i", original_video_in_output, "-ac", "1", "-ar", "16000", temp_audio], check=True, capture_output=True)

    # 2. Transcribe (Lip Sync mode)
    print(f"🧠 Transcribing for Lip Sync...")
    output_base = os.path.join(output_dir, "whisper_out")
    
    cmd = [
        WHISPER_CLI,
        "-m", model_path,
        "-f", temp_audio,
        "-ojf", "-osrt",
        "-of", output_base,
        "-l", "auto",
        "-t", "8",
        "-bs", "5",
        "-ml", "1",
        "-sow"
    ]
    cmd.extend(get_whisper_gpu_args())
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Transcription failed: {result.stderr}")
        return

    # 3. Parse and Store SRTs
    srt_out_path = output_base + ".srt"
    if os.path.exists(srt_out_path):
        segments = parse_srt_to_segments(srt_out_path)
    else:
        print("❌ Error: No subtitle output found.")
        print(f"DEBUG - stderr: {result.stderr}")
        print(f"DEBUG - stdout: {result.stdout}")
        return
    
    # Merge for display
    merged_segments = merge_segments(segments, max_words=args.max_words)
    
    # Generate Dialogue (Simple)
    dialogue_srt = generate_srt_content(merged_segments, use_roman=args.roman, use_karaoke=False, max_words_per_line=args.max_words)
    dialogue_path = os.path.join(output_dir, "dialogue.srt")
    with open(dialogue_path, "w", encoding="utf-8") as f:
        f.write(dialogue_srt)
        
    # Generate Karaoke (Lip Sync)
    karaoke_srt = generate_srt_content(merged_segments, use_roman=args.roman, use_karaoke=True, highlight_color=args.highlight_color, max_words_per_line=args.max_words)
    karaoke_path = os.path.join(output_dir, "lip_sync.srt")
    with open(karaoke_path, "w", encoding="utf-8") as f:
        f.write(karaoke_srt)

    # 4. Burn Video
    print("🔥 Burning Final Video...")
    final_video_path = os.path.join(output_dir, "final_video.mov")
    
    def hex_to_ass(hex_c):
        hex_c = hex_c.lstrip('#')
        return f"&H00{hex_c[4:6]}{hex_c[2:4]}{hex_c[0:2]}"
    
    ass_base = hex_to_ass("#FFFFFF")
    tech_font = AVAILABLE_FONTS.get(args.base_font, 'Arial')
    
    style = f"FontName={tech_font},FontSize={args.font_size},PrimaryColour={ass_base},OutlineColour=&H000000,BorderStyle=1,Outline=2,Shadow=0,MarginV=50,Alignment=2"
    filter_str = f"subtitles=filename='{karaoke_path}':force_style='{style}'"
    
    encoder = get_best_ffmpeg_encoder()
    cmd_burn = [
        FFMPEG_PATH, "-y", 
        "-i", original_video_in_output, 
        "-vf", filter_str,
        "-c:v", encoder, "-b:v", "8M",
        "-c:a", "copy",
        final_video_path
    ]
    
    if encoder == "h264_videotoolbox":
        cmd_burn.insert(10, "-realtime")
        cmd_burn.insert(11, "1")
    
    try:
        subprocess.run(cmd_burn, check=True)
        print(f"\n✨ ALL DONE! Organized files saved in: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg Error: {e}")

if __name__ == "__main__":
    main()
